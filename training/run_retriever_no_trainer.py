import argparse
import functools
import logging
import math
from random import choice, randint

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils import checkpoint
from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import get_scheduler, AutoTokenizer, AdamW, SchedulerType, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Train ELI5 retriever")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vblagoje/lfqa",
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1024,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--checkpoint_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="google/bert_uncased_L-8_H-768_A-12",
    )

    parser.add_argument(
        "--model_save_name",
        type=str,
        default="eli5_retriever_model_l-12_h-768_b-512-512",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.2,
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


class RetrievalQAEmbedder(torch.nn.Module):
    def __init__(self, sent_encoder):
        super(RetrievalQAEmbedder, self).__init__()
        dim = sent_encoder.config.hidden_size
        self.bert_query = sent_encoder
        self.output_dim = 128
        self.project_query = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.project_doc = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def embed_sentences_checkpointed(self, input_ids, attention_mask, checkpoint_batch_size=-1):
        # reproduces BERT forward pass with checkpointing
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            return self.bert_query(input_ids, attention_mask=attention_mask)[1]
        else:
            # prepare implicit variables
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * self.bert_query.config.num_hidden_layers
            extended_attention_mask: torch.Tensor = self.bert_query.get_extended_attention_mask(
                attention_mask, input_shape, device
            )

            # define function for checkpointing
            def partial_encode(*inputs):
                encoder_outputs = self.bert_query.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask, )
                sequence_output = encoder_outputs[0]
                pooled_output = self.bert_query.pooler(sequence_output)
                return pooled_output

            # run embedding layer on everything at once
            embedding_output = self.bert_query.embeddings(
                input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
            )
            # run encoding and pooling on one mini-batch at a time
            pooled_output_list = []
            for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * checkpoint_batch_size: (b + 1) * checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * checkpoint_batch_size: (b + 1) * checkpoint_batch_size]
                pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
                pooled_output_list.append(pooled_output)
            return torch.cat(pooled_output_list, dim=0)

    def embed_questions(self, q_ids, q_mask, checkpoint_batch_size=-1):
        q_reps = self.embed_sentences_checkpointed(q_ids, q_mask, checkpoint_batch_size)
        return self.project_query(q_reps)

    def embed_answers(self, a_ids, a_mask, checkpoint_batch_size=-1):
        a_reps = self.embed_sentences_checkpointed(a_ids, a_mask, checkpoint_batch_size)
        return self.project_doc(a_reps)

    def forward(self, q_ids, q_mask, a_ids, a_mask, checkpoint_batch_size=-1):
        device = q_ids.device
        q_reps = self.embed_questions(q_ids, q_mask, checkpoint_batch_size)
        a_reps = self.embed_answers(a_ids, a_mask, checkpoint_batch_size)
        compare_scores = torch.mm(q_reps, a_reps.t())
        loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.shape[1]).to(device))
        loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.shape[0]).to(device))
        loss = (loss_qa + loss_aq) / 2
        return loss


class ELI5DatasetQARetriever(Dataset):
    def __init__(self, examples_array, extra_answer_threshold=3, min_answer_length=64, training=True, n_samples=None):
        self.data = examples_array
        self.answer_thres = extra_answer_threshold
        self.min_length = min_answer_length
        self.training = training
        self.n_samples = self.data.num_rows if n_samples is None else n_samples

    def __len__(self):
        return self.n_samples

    def make_example(self, idx):
        example = self.data[idx]
        question = example["title"]
        if self.training:
            answers = [a for i, (a, sc) in enumerate(zip(example["answers"]["text"], example["answers"]["score"]))]
            answer_tab = choice(answers).split(" ")
            start_idx = randint(0, max(0, len(answer_tab) - self.min_length))
            answer_span = " ".join(answer_tab[start_idx:])
        else:
            answer_span = example["answers"]["text"][0]
        return question, answer_span

    def __getitem__(self, idx):
        return self.make_example(idx % self.data.num_rows)


def make_qa_retriever_batch(qa_list, tokenizer, max_len=64):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer(q_ls, padding="max_length", max_length=max_len, truncation=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]),
        torch.LongTensor(q_toks["attention_mask"])
    )
    a_toks = tokenizer(a_ls, padding="max_length", max_length=max_len, truncation=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]),
        torch.LongTensor(a_toks["attention_mask"]),
    )
    return q_ids, q_mask, a_ids, a_mask


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
    data_files = {"train": "train.json", "validation": "validation.json", "test": "test.json"}
    eli5 = load_dataset(args.dataset_name, data_files=data_files)

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
    train_dataset = ELI5DatasetQARetriever(eli5['train'], training=True)
    valid_dataset = ELI5DatasetQARetriever(eli5['validation'], training=False)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    base_model = AutoModel.from_pretrained(args.pretrained_model_name)

    model = RetrievalQAEmbedder(base_model)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    model_collate_fn = functools.partial(make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                                  sampler=RandomSampler(train_dataset), collate_fn=model_collate_fn)

    model_collate_fn = functools.partial(make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size,
                                 sampler=SequentialSampler(valid_dataset), collate_fn=model_collate_fn)

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
        batch = next(iter(train_dataloader))
        for step in range(1000):
        #for step, batch in enumerate(train_dataloader, start=checkpoint_step):
            # model inputs
            q_ids, q_mask, a_ids, a_mask = batch
            pre_loss = model(q_ids, q_mask, a_ids, a_mask, checkpoint_batch_size=args.checkpoint_batch_size)
            loss = pre_loss.sum() / args.gradient_accumulation_steps
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
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                eval_loss = evaluate_qa_retriever(unwrapped_model, eval_dataloader)
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

