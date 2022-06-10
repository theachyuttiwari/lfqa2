import argparse
import json

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DPRQuestionEncoder

from common import embed_questions, clean_question, articles_to_paragraphs, kilt_wikipedia_columns
from common import kilt_wikipedia_paragraph_columns as columns


def generate_dpr_training_file(args):
    n_negatives = 7
    min_chars_per_passage = 200

    def query_index(question, topk=(n_negatives * args.n_positives) * 2):
        question_embedding = embed_questions(question_model, question_tokenizer, [question])
        scores, wiki_passages = kilt_wikipedia_paragraphs.get_nearest_examples("embeddings", question_embedding, k=topk)

        retrieved_examples = []
        r = list(zip(wiki_passages[k] for k in columns))
        for i in range(topk):
            retrieved_examples.append({k: v for k, v in zip(columns, [r[j][0][i] for j in range(len(columns))])})

        return retrieved_examples

    def find_positive_and_hard_negative_ctxs(dataset_index: int, n_positive=1, device="cuda:0"):
        positive_context_list = []
        hard_negative_context_list = []
        example = dataset[dataset_index]
        question = clean_question(example['title'])
        passages = query_index(question)
        passages = [dict([(k, p[k]) for k in columns]) for p in passages]
        q_passage_pairs = [[question, f"{p['title']} {p['text']}" if args.use_title else p["text"]] for p in passages]

        features = ce_tokenizer(q_passage_pairs, padding="max_length", max_length=256, truncation=True,
                                return_tensors="pt")
        with torch.no_grad():
            passage_scores = ce_model(features["input_ids"].to(device),
                                      features["attention_mask"].to(device)).logits

        for p_idx, p in enumerate(passages):
            p["score"] = passage_scores[p_idx].item()

        # order by scores
        def score_passage(item):
            return item["score"]

        # pick the most relevant as the positive answer
        best_passage_list = sorted(passages, key=score_passage, reverse=True)
        for idx, item in enumerate(best_passage_list):
            if idx < n_positive:
                positive_context_list.append({"title": item["title"], "text": item["text"]})
            else:
                break

        # least relevant as hard_negative
        worst_passage_list = sorted(passages, key=score_passage, reverse=False)
        for idx, hard_negative in enumerate(worst_passage_list):
            if idx < n_negatives * n_positive:
                hard_negative_context_list.append({"title": hard_negative["title"], "text": hard_negative["text"]})
            else:
                break
        assert len(positive_context_list) * n_negatives == len(hard_negative_context_list)
        return positive_context_list, hard_negative_context_list

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    question_model = DPRQuestionEncoder.from_pretrained(args.question_encoder_name).to(device)
    question_tokenizer = AutoTokenizer.from_pretrained(args.question_encoder_name)
    _ = question_model.eval()

    ce_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-4-v2').to(device)
    ce_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-4-v2')
    _ = ce_model.eval()

    kilt_wikipedia = load_dataset("kilt_wikipedia", split="full")

    kilt_wikipedia_paragraphs = kilt_wikipedia.map(articles_to_paragraphs, batched=True,
                                                   remove_columns=kilt_wikipedia_columns,
                                                   batch_size=512,
                                                   cache_file_name=f"../data/wiki_kilt_paragraphs_full.arrow",
                                                   desc="Expanding wiki articles into paragraphs")

    # use paragraphs that are not simple fragments or very short sentences
    # Wikipedia Faiss index needs to fit into a 16 Gb GPU
    kilt_wikipedia_paragraphs = kilt_wikipedia_paragraphs.filter(
        lambda x: (x["end_character"] - x["start_character"]) > min_chars_per_passage)

    kilt_wikipedia_paragraphs.load_faiss_index("embeddings", args.index_file_name, device=0)

    eli5_train_set = load_dataset("vblagoje/lfqa", split="train")
    eli5_validation_set = load_dataset("vblagoje/lfqa", split="validation")
    eli5_test_set = load_dataset("vblagoje/lfqa", split="test")

    for dataset_name, dataset in zip(["train", "validation", "test"], [eli5_train_set,
                                                                       eli5_validation_set,
                                                                       eli5_test_set]):

        progress_bar = tqdm(range(len(dataset)), desc=f"Creating DPR formatted {dataset_name} file")
        with open('eli5-dpr-' + dataset_name + '.jsonl', 'w') as fp:
            for idx, example in enumerate(dataset):
                negative_start_idx = 0
                positive_context, hard_negative_ctxs = find_positive_and_hard_negative_ctxs(idx, args.n_positives,
                                                                                            device)
                for pc in positive_context:
                    hnc = hard_negative_ctxs[negative_start_idx:negative_start_idx + n_negatives]
                    json.dump({"id": example["q_id"],
                               "question": clean_question(example["title"]),
                               "positive_ctxs": [pc],
                               "hard_negative_ctxs": hnc}, fp)
                    fp.write("\n")
                    negative_start_idx += n_negatives
                progress_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates DPR training file")
    parser.add_argument(
        "--use_title",
        action="store_true",
        help="If true, use title in addition to passage text for passage embedding",
    )
    parser.add_argument(
        "--n_positives",
        default=3,
        help="Number of positive samples per question",
    )
    parser.add_argument(
        "--question_encoder_name",
        default="vblagoje/dpr-question_encoder-single-lfqa-base",
        help="Question encoder to use",
    )

    parser.add_argument(
        "--index_file_name",
        default="../data/kilt_dpr_wikipedia_first.faiss",
        help="Faiss index with passage embeddings",
    )

    main_args, _ = parser.parse_known_args()
    generate_dpr_training_file(main_args)
