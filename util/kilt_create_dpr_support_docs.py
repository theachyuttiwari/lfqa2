import argparse
import json
import os

import faiss
import torch
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder

from common import articles_to_paragraphs, embed_questions, embed_passages, create_kilt_datapoint, \
    kilt_wikipedia_columns
from common import kilt_wikipedia_paragraph_columns as columns


def generate_support_docs(args):
    dims = 128
    min_chars_per_passage = 200
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    lfqa = load_dataset("vblagoje/lfqa")

    ctx_tokenizer = AutoTokenizer.from_pretrained(args.ctx_encoder_name)
    ctx_model = DPRContextEncoder.from_pretrained(args.ctx_encoder_name).to(device)
    _ = ctx_model.eval()

    question_tokenizer = AutoTokenizer.from_pretrained(args.question_encoder_name)
    question_model = DPRQuestionEncoder.from_pretrained(args.question_encoder_name).to(device)
    _ = question_model.eval()

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

    def query_index(question, topk=7):
        topk = topk * 3  # grab 3x results and filter for word count
        question_embedding = embed_questions(question_model, question_tokenizer, [question])
        scores, wiki_passages = kilt_wikipedia_paragraphs.get_nearest_examples("embeddings", question_embedding, k=topk)

        retrieved_examples = []
        r = list(zip(wiki_passages[k] for k in columns))
        for i in range(topk):
            retrieved_examples.append({k: v for k, v in zip(columns, [r[j][0][i] for j in range(len(columns))])})

        return retrieved_examples

    def create_support_doc(dataset: Dataset, output_filename: str):
        progress_bar = tqdm(range(len(dataset)), desc="Creating supporting docs")

        with open(output_filename, "w") as fp:
            for example in dataset:
                wiki_passages = query_index(example["title"])
                kilt_dp = create_kilt_datapoint(example, columns, wiki_passages)
                json.dump(kilt_dp, fp)
                fp.write("\n")
                progress_bar.update(1)

    if not os.path.isfile(args.index_file_name):
        def embed_passages_for_retrieval(examples):
            return embed_passages(ctx_model, ctx_tokenizer, examples, max_length=128)

        paragraphs_embeddings = kilt_wikipedia_paragraphs.map(embed_passages_for_retrieval,
                                                              batched=True, batch_size=512,
                                                              cache_file_name=args.encoded_kilt_file_name,
                                                              desc="Creating faiss index")

        paragraphs_embeddings.add_faiss_index(column="embeddings", custom_index=faiss.IndexFlatIP(dims))
        paragraphs_embeddings.save_faiss_index("embeddings", args.index_file_name)

    kilt_wikipedia_paragraphs.load_faiss_index("embeddings", args.index_file_name, device=0)
    create_support_doc(lfqa["train"], "lfqa_dpr_train_precomputed_dense_docs.json")
    create_support_doc(lfqa["validation"], "lfqa_dpr_validation_precomputed_dense_docs.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates support docs for seq2seq model training")
    parser.add_argument(
        "--ctx_encoder_name",
        default="vblagoje/dpr-ctx_encoder-single-lfqa-base",
        help="Question encoder to use",
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

    parser.add_argument(
        "--encoded_kilt_file_name",
        default="../data/kilt_embedded.arrow",
        help="Encoded KILT file name",
    )

    main_args, _ = parser.parse_known_args()
    generate_support_docs(main_args)
