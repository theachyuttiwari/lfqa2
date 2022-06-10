import argparse
import logging
import math
import re

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler, AutoTokenizer, AdamW, SchedulerType, AutoModelForSeq2SeqLM, \
    DataCollatorWithPadding

from datasets import load_dataset

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Train ELI5 seq2seq answer generation model")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vblagoje/lfqa",
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="facebook/bart-large",
    )

    parser.add_argument(
        "--model_save_name",
        type=str,
        default="eli5_bart_model",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use."
    )

    parser.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="Log train/validation loss every log_freq update steps"
    )

    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
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
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
             "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_target_length",
        type=int,
        default=360,
        help="The maximum total sequence length for target text after "
             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
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
        default=None,
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--warmup_percentage",
        type=float,
        default=0.08,
        help="Number of steps for the warmup in the lr scheduler."
    )
    return parser


def cleanup_references(text):
    # URL reference where we need to remove both the link text and URL
    # ...and this letter is used by most biographers as the cornerstone of Lee's personal
    # views on slavery ([1](_URL_2_ & pg=PA173), [2](_URL_1_), [3](_URL_5_)).
    # ...and this letter is used by most biographers as the cornerstone of Lee's personal views on slavery.
    result = re.sub(r"[\(\s]*\[\d+\]\([^)]+\)[,)]*", "", text, 0, re.MULTILINE)

    # URL reference where we need to preserve link text but remove URL
    # At the outbreak of the Civil War, [Leyburn left his church](_URL_19_) and joined the South.
    # At the outbreak of the Civil War, Leyburn left his church and joined the South.
    result = re.sub(r"\[([^]]+)\]\([^)]+\)", "\\1", result, 0, re.MULTILINE)

    # lastly remove just dangling _URL_[0-9]_ URL references
    result = re.sub(r"_URL_\d_", "", result, 0, re.MULTILINE)
    return result


def clean_answer(text):
    result = cleanup_references(text)
    result = result.replace("\n", " ")
    result = re.sub(r"\s\s+", " ", result)
    result = re.sub(r"BULLET::::-", "", result)
    return result.strip()


def clean_question(text):
    result = cleanup_references(text)
    result = result.replace("\n", " ")
    result = re.sub(r"\s\s+", " ", result)
    result = result.replace("[deleted]", "")
    return result.lower().strip()


def prepare_support_docs(example):
    provenances = example["output"][-1]["provenance"]
    context = "<P> " + " <P> ".join([p["text"] for p in provenances])
    return {"context": context}


def preprocess_eli5(examples, **fn_kwargs):
    document_cache = fn_kwargs["document_cache"]
    training = fn_kwargs.get("training", True)
    extra_answer_threshold = fn_kwargs.get("extra_answer_threshold", 3)
    include_selftext = fn_kwargs.get("include_selftext", False)
    exclude_answer_patterns = fn_kwargs.get("exclude_answer_patterns", [])

    questions, contexts, answers = [], [], []
    for q_id, question, selftext, answer in zip(examples["q_id"], examples["title"], examples["selftext"],
                                                examples["answers"]):
        accepted_answer_idx = []
        if training:
            accepted_answer_idx = [idx for idx, score in enumerate(answer["score"]) if
                                   score > extra_answer_threshold]
        if not training or not accepted_answer_idx:
            accepted_answer_idx = [0]
        document = document_cache[q_id]
        for idx in accepted_answer_idx:
            skip_answer = any([p.search(answer["text"][idx]) for p in exclude_answer_patterns])
            if skip_answer:
                continue
            if include_selftext:
                questions.append(clean_question(f"{question} {selftext}"))
            else:
                questions.append(clean_question(question))
            contexts.append(document.lower().strip())
            answers.append(clean_answer(answer["text"][idx]))

    return {"question": questions, "context": contexts, "answer": answers}


def eval_qa_s2s_epoch(model, dataloader, accelerator, args):
    model.eval()
    num_eval_steps = math.ceil(len(dataloader))
    progress_bar = tqdm(range(num_eval_steps), disable=not accelerator.is_local_main_process)
    total_loss = 0.
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=round((total_loss / (step + 1)), 3))
        return total_loss / (step + 1)


def train(config):
    set_seed(42)
    args = config["args"]
    eli5 = load_dataset(args.dataset_name)

    support_docs = load_dataset("vblagoje/lfqa_support_docs")

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

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    processed_datasets = {}
    support_docs_prepared = {}
    with accelerator.main_process_first():
        for split in ["train", "validation"]:
            support_docs_prepared[split] = support_docs[split].map(prepare_support_docs,
                                                                   batched=False,
                                                                   cache_file_name=f"./support_docs_{split}.arrow",
                                                                   load_from_cache_file=not args.overwrite_cache,
                                                                   desc="Preparing support docs",
                                                                   )
        column_names = eli5["train"].column_names
        for split in ["train", "validation"]:
            d_cache = dict([(e["id"], e["context"]) for e in tqdm(support_docs_prepared[split],
                                                                  desc=f"Adding support docs to LFQA {split}")])
            processed_datasets[split] = eli5[split].map(preprocess_eli5,
                                                        batched=True,
                                                        remove_columns=column_names,
                                                        cache_file_name=f"./processed_datasets_{split}.arrow",
                                                        load_from_cache_file=not args.overwrite_cache,
                                                        desc="Preparing dataset for tokenization",
                                                        fn_kwargs={"document_cache": d_cache,
                                                                   "training": split == "train",
                                                                   "exclude_answer_patterns": [re.compile("not sure what you"),
                                                                                               re.compile("\n\n >")]}
                                                        )

    padding = "max_length" if args.pad_to_max_length else False
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    def tokenize_dataset(examples):
        inputs = ["question: {} context: {}".format(q, c) for q, c in zip(examples["question"], examples["context"])]
        targets = examples["answer"]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=True, truncation=True,
                               return_tensors="np")

        model_inputs["decoder_input_ids"] = labels["input_ids"][:, :-1].tolist()
        # replace pad_token_id with label_pad_token_id to avoid loss calculation on those tokens
        labels["input_ids"] = np.where(labels["input_ids"] == tokenizer.pad_token_id,
                                       label_pad_token_id, labels["input_ids"])

        model_inputs["labels"] = labels["input_ids"][:, 1:].tolist()
        return model_inputs

    tokenized_datasets = {}
    with accelerator.main_process_first():
        for split, dataset in processed_datasets.items():
            tokenized_datasets[split] = dataset.map(
                tokenize_dataset,
                batched=True,
                cache_file_name=f"./tokenized_dataset_{split}.arrow",
                remove_columns=dataset.column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    train_dataset.set_format(type='torch')
    eval_dataset.set_format(type='torch')

    data_collator = DataCollatorWithPadding(tokenizer, "max_length")

    # first epoch we don't shuffle
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.per_device_train_batch_size,
                                  collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator)

    # train the model
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              eval_dataloader)
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
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num eval examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Warmup steps = {num_warmup_steps}")
    logger.info(f"  Logging training progress every {args.log_freq} optimization steps")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    switched_train_dataloader = False
    for epoch in range(args.num_train_epochs):
        model.train()
        if epoch > 0 and not switched_train_dataloader:
            train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                                          shuffle=True, collate_fn=data_collator)
            train_dataloader = accelerator.prepare(train_dataloader)
            switched_train_dataloader = True

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = torch.mean(outputs.loss)
            accelerator.backward(loss)
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(train_dataloader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(loss=round(loss.item(), 3))
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            if step % (args.log_freq * args.gradient_accumulation_steps) == 0:
                validation_loss = eval_qa_s2s_epoch(model, eval_dataloader, accelerator, args)
                model.train()
                logger.info(f"Train loss {loss.item()} , validation loss {validation_loss}")
                if args.wandb and accelerator.is_local_main_process:
                    import wandb
                    wandb.log({"loss": loss.item(),
                               "lr": scheduler.get_last_lr()[0],
                               "validation_loss": validation_loss,
                               "completed_steps": completed_steps})

        logger.info("Saving model {}".format(args.model_save_name))
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), "{}_{}.bin".format(args.model_save_name, epoch))

        # Calculating the validation loss over epoch
        validation_loss = eval_qa_s2s_epoch(model, eval_dataloader, accelerator, args)

        logger.info("Epoch: {}".format(epoch))
        logger.info("Validation loss: {}".format(validation_loss))


def main():
    parser = get_parser()
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If true, use W&B logging",
    )
    main_args, _ = parser.parse_known_args()
    config = {"args": main_args}
    if main_args.wandb:
        import wandb
        wandb.init(project="Bart_ELI5")
    train(config=config)


main()



