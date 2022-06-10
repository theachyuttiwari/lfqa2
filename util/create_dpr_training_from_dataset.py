import argparse
import random
import json
import re

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, cos_sim
from tqdm.auto import tqdm
from datasets import load_dataset

from common import clean_answer, clean_question


def find_hard_negative_ctxs(dataset, dataset_embeddings, embedding_index: int,
                            exclude_answer_patterns, similarity_threshold=[0.5, 0.6], k=25, min_count=3):
    hard_negative_ctxs = []
    results = semantic_search(dataset_embeddings[embedding_index], dataset_embeddings, top_k=k,
                              score_function=cos_sim)
    # list if dicts
    # [{'corpus_id': 8, 'score': -0.019427383318543434},
    #  ...
    # {'corpus_id': 10, 'score': -0.09040290117263794}]
    # hard negative are most similar and negatives are most disimilar to embedding_index
    hard_negative_results = results[0][1:k + 1]
    assert len(hard_negative_results) > min_count * 2
    for r in hard_negative_results:
        example = dataset[r["corpus_id"]]
        if similarity_threshold[0] < r["score"] <= similarity_threshold[1]:
            for a in example["answers"]["text"]:
                hard_negative_ctxs.append({"title": "", "text": clean_answer(a)})
        if len(hard_negative_ctxs) > min_count:
            break
    return hard_negative_ctxs[:min_count]


def find_negative_ctxs(dataset, dataset_embeddings, embedding_index: int,
                       exclude_answer_patterns, similarity_threshold=0.1, k=7, min_count=3):
    negative_ctxs = []
    random_sample = random.sample(range(len(dataset_embeddings)), k * 20)
    similarities = cos_sim(dataset_embeddings[embedding_index], dataset_embeddings[random_sample])[0].tolist()
    for idx, score in enumerate(similarities):
        if score < similarity_threshold:
            example = dataset[random_sample[idx]]
            for a in example["answers"]["text"]:
                negative_ctxs.append({"title": "", "text": clean_answer(a)})
        if len(negative_ctxs) > min_count:
            break
    return negative_ctxs[:min_count]


def generate_dpr_training_file(args):
    embedder = SentenceTransformer(args.embedding_model)

    eli5_train_set = load_dataset("vblagoje/lfqa", split="train")
    eli5_validation_set = load_dataset("vblagoje/lfqa", split="validation")
    eli5_test_set = load_dataset("vblagoje/lfqa", split="test")

    train_set = embedder.encode([example["title"] for example in eli5_train_set], convert_to_tensor=True,
                                show_progress_bar=True)
    validation_set = embedder.encode([example["title"] for example in eli5_validation_set], convert_to_tensor=True,
                                     show_progress_bar=True)

    test_set = embedder.encode([example["title"] for example in eli5_test_set], convert_to_tensor=True,
                               show_progress_bar=True)
    exclude_answer_patterns = [re.compile("not sure what you"), re.compile("\n\n >")]
    for dataset_name, dataset, dataset_embeddings in zip(["train", "validation", "test"],
                                                         [eli5_train_set, eli5_validation_set, eli5_test_set],
                                                         [train_set, validation_set, test_set]):
        min_elements = 3
        skip_count = 0
        progress_bar = tqdm(range(len(dataset)), desc="Creating DPR formatted question/passage docs")
        with open('eli5-dpr-' + dataset_name + '.jsonl', 'w') as fp:
            for idx, example in enumerate(dataset):
                negative_ctxs = find_negative_ctxs(dataset, dataset_embeddings, idx, exclude_answer_patterns)
                hard_negative_ctxs = find_hard_negative_ctxs(dataset, dataset_embeddings, idx, exclude_answer_patterns)
                positive_context = [{"text": clean_answer(a), "title": ""} for a in example["answers"]["text"] if
                                    not any([p.search(a) for p in exclude_answer_patterns])]
                if not positive_context:
                    positive_context = [{"text": clean_answer(a), "title": ""} for a in example["answers"]["text"]]
                if len(positive_context) > 0 and len(negative_ctxs) > 0 and len(hard_negative_ctxs) >= min_elements:
                    json.dump({"id": example["q_id"],
                               "question": clean_question(example["title"]),
                               "positive_ctxs": positive_context[:min_elements],
                               "negative_ctxs": negative_ctxs[:min_elements],
                               "hard_negative_ctxs": hard_negative_ctxs[:min_elements]}, fp)
                    fp.write("\n")
                else:
                    skip_count += 1
                progress_bar.update(1)

        print(f"Skipped {skip_count} questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates DPR training file from LFQA dataset")
    parser.add_argument(
        "--embedding_model",
        default="all-mpnet-base-v2",
        help="Embedding model to use for question encoding and semantic search",
    )

    main_args, _ = parser.parse_known_args()
    generate_dpr_training_file(main_args)
