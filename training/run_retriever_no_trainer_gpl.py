import argparse
import logging
import math
from dataclasses import dataclass
from typing import List, Any, Union, Optional

import torch
import ujson
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import get_scheduler, AutoTokenizer, AutoModel, AdamW, SchedulerType, PreTrainedTokenizerBase, AutoModelForSequenceClassification, BatchEncoding
from transformers.file_utils import PaddingStrategy

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Train LFQA retriever")
    parser.add_argument(
        "--dpr_input_file",
        type=str,
        help="DPR formatted input file with question/positive/negative pairs in a JSONL file",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )


    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--ce_model_name",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    parser.add_argument(
        "--model_save_name",
        type=str,
        default="eli5_retriever_model_l-12_h-768_b-512-512",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--log_freq",
        type=int,
        default=500,
        help="Log train/validation loss every log_freq update steps"
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",  # this is linear with warmup
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--warmup_percentage",
        type=float,
        default=0.08,
        help="Number of steps for the warmup in the lr scheduler."
    )
    return parser


@dataclass
class InputExample:
    guid: str = ""
    texts: List[str] = None
    label: Union[int, float] = 0


class DPRDataset(Dataset):
    """
    Dataset DPR format of question, answers, positive, negative, and hard negative passages
    See https://github.com/facebookresearch/DPR#retriever-input-data-format for more details
    """

    def __init__(self, file_path: str, include_all_positive: bool = False) -> None:
        super().__init__()
        with open(file_path, "r") as fp:
            self.data = []

            def dpr_example_to_input_example(idx, dpr_item):
                examples = []
                for p_idx, p_item in enumerate(dpr_item["positive_ctxs"]):
                    for n_idx, n_item in enumerate(dpr_item["negative_ctxs"]):
                        examples.append(InputExample(guid=[idx, p_idx, n_idx], texts=[dpr_item["question"],
                                                                                      p_item["text"],
                                                                                      n_item["text"]]))
                    if not include_all_positive:
                        break
                return examples

            for idx, line in enumerate(fp):
                self.data.extend(dpr_example_to_input_example(idx, ujson.loads(line)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def dpr_collate_fn(batch):
    query_id, pos_id, neg_id = zip(*[example.guid for example in batch])
    query, pos, neg = zip(*[example.texts for example in batch])
    return (query_id, pos_id, neg_id), (query, pos, neg)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


@dataclass
class CrossEncoderCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Any
    target_tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, batch):
        query_id, pos_id, neg_id = zip(*[example.guid for example in batch])
        query, pos_passage, neg_passage = zip(*[example.texts for example in batch])
        batch_input: List[List[str]] = list(zip(query, pos_passage)) + list(zip(query, neg_passage))
        features = self.tokenizer(batch_input, padding=self.padding, truncation=True,
                                  return_tensors=self.return_tensors)
        with torch.no_grad():
            scores = self.model(**features).logits

        labels = scores[:len(query)] - scores[len(query):]
        batch_input: List[str] = list(query) + list(pos_passage) + list(neg_passage)
        #breakpoint()
        encoded_input = self.target_tokenizer(batch_input, padding=True, truncation=True,
                                              max_length=256, return_tensors='pt')

        encoded_input["labels"] = labels

        return encoded_input


class RetrievalQAEmbedder(torch.nn.Module):
    def __init__(self, sent_encoder, sent_tokenizer, batch_size:int = 32):
        super(RetrievalQAEmbedder, self).__init__()
        dim = sent_encoder.config.hidden_size
        self.model = sent_encoder
        self.tokenizer = sent_tokenizer
        self.scale = 1
        self.similarity_fct = 'dot'
        self.batch_size = 32
        self.loss_fct = nn.MSELoss()

    def forward(self, examples: BatchEncoding):
        # Tokenize sentences
        labels = examples.pop("labels")
        # Compute token embeddings
        model_output = self.model(**examples)

        examples["labels"] = labels

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, examples['attention_mask'])
        target_shape = (3, self.batch_size, sentence_embeddings.shape[-1])
        sentence_embeddings_reshaped = torch.reshape(sentence_embeddings, target_shape)
        
        #breakpoint()

        embeddings_query = sentence_embeddings_reshaped[0]
        embeddings_pos = sentence_embeddings_reshaped[1]
        embeddings_neg = sentence_embeddings_reshaped[2]

        if self.similarity_fct == 'cosine':
            embeddings_query = F.normalize(embeddings_query, p=2, dim=1)
            embeddings_pos = F.normalize(embeddings_pos, p=2, dim=1)
            embeddings_neg = F.normalize(embeddings_neg, p=2, dim=1)

        scores_pos = (embeddings_query * embeddings_pos).sum(dim=-1) * self.scale
        scores_neg = (embeddings_query * embeddings_neg).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg
        #breakpoint()
        return self.loss_fct(margin_pred, labels.squeeze())


def evaluate_qa_retriever(model, data_loader):
    # make iterator
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    tot_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            q_ids, q_mask, a_ids, a_mask = batch
            loss = model(q_ids, q_mask, a_ids, a_mask)
            tot_loss += loss.item()
        return tot_loss / (step + 1)


def train(config):
    set_seed(42)
    args = config["args"]

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    logger.info(accelerator.state)

    # prepare torch Dataset objects
    train_dataset = DPRDataset(file_path=args.dpr_input_file)
    valid_dataset = Dataset()

    base_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    base_model = AutoModel.from_pretrained(args.pretrained_model_name)

    ce_tokenizer = AutoTokenizer.from_pretrained(args.ce_model_name)
    ce_model = AutoModelForSequenceClassification.from_pretrained(args.ce_model_name)
    _ = ce_model.eval()

    model = RetrievalQAEmbedder(base_model, base_tokenizer)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    cec = CrossEncoderCollator(model=ce_model, tokenizer=ce_tokenizer, target_tokenizer=base_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                                  sampler=RandomSampler(train_dataset), collate_fn=cec)

    eval_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size,
                                 sampler=SequentialSampler(valid_dataset), collate_fn=cec)

    # train the model
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer,
                                                                              train_dataloader, eval_dataloader)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    num_warmup_steps = args.num_warmup_steps if args.num_warmup_steps else math.ceil(args.max_train_steps *
                                                                                     args.warmup_percentage)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Warmup steps = {num_warmup_steps}")
    logger.info(f"  Logging training progress every {args.log_freq} optimization steps")

    loc_loss = 0.0
    current_loss = 0.0
    checkpoint_step = 0

    completed_steps = checkpoint_step
    progress_bar = tqdm(range(args.max_train_steps), initial=checkpoint_step,
                        disable=not accelerator.is_local_main_process)
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader, start=checkpoint_step):
            # model inputs
            pre_loss = model(batch)
            loss = pre_loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            loc_loss += loss.item()
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(train_dataloader)):
                current_loss = loc_loss
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loc_loss)
                loc_loss = 0
                completed_steps += 1

            if step % (args.log_freq * args.gradient_accumulation_steps) == 0:
                # accelerator.wait_for_everyone()
                # unwrapped_model = accelerator.unwrap_model(model)
                # eval_loss = evaluate_qa_retriever(unwrapped_model, eval_dataloader)
                eval_loss = 0
                logger.info(f"Train loss {current_loss} , eval loss {eval_loss}")
                if args.wandb and accelerator.is_local_main_process:
                    import wandb
                    wandb.log({"loss": current_loss, "eval_loss": eval_loss, "step": completed_steps})

            if completed_steps >= args.max_train_steps:
                break

        logger.info("Saving model {}".format(args.model_save_name))
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), "{}_{}.bin".format(args.model_save_name, epoch))
        eval_loss = evaluate_qa_retriever(unwrapped_model, eval_dataloader)
        logger.info("Evaluation loss epoch {:4d}: {:.3f}".format(epoch, eval_loss))


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use W&B logging",
    )
    main_args, _ = parser.parse_known_args()
    config = {"args": main_args}
    if main_args.wandb:
        import wandb

        wandb.init(project="Retriever")

    train(config=config)

