import argparse
import os

import faiss
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DPRContextEncoder

from common import articles_to_paragraphs, embed_passages


def create_faiss(args):
    dims = 128
    min_chars_per_passage = 200
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    ctx_tokenizer = AutoTokenizer.from_pretrained(args.ctx_encoder_name)
    ctx_model = DPRContextEncoder.from_pretrained(args.ctx_encoder_name).to(device)
    _ = ctx_model.eval()

    kilt_wikipedia = load_dataset("kilt_wikipedia", split="full")
    kilt_wikipedia_columns = ['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text', 'anchors', 'categories',
                              'wikidata_info', 'history']

    kilt_wikipedia_paragraphs = kilt_wikipedia.map(articles_to_paragraphs, batched=True,
                                                   remove_columns=kilt_wikipedia_columns,
                                                   batch_size=512,
                                                   cache_file_name=f"../data/wiki_kilt_paragraphs_full.arrow",
                                                   desc="Expanding wiki articles into paragraphs")

    # use paragraphs that are not simple fragments or very short sentences
    # Wikipedia Faiss index needs to fit into a 16 Gb GPU
    kilt_wikipedia_paragraphs = kilt_wikipedia_paragraphs.filter(
        lambda x: (x["end_character"] - x["start_character"]) > min_chars_per_passage)

    if not os.path.isfile(args.index_file_name):
        def embed_passages_for_retrieval(examples):
            return embed_passages(ctx_model, ctx_tokenizer, examples, max_length=128)

        paragraphs_embeddings = kilt_wikipedia_paragraphs.map(embed_passages_for_retrieval,
                                                              batched=True, batch_size=512,
                                                              cache_file_name="../data/kilt_embedded.arrow",
                                                              desc="Creating faiss index")

        paragraphs_embeddings.add_faiss_index(column="embeddings", custom_index=faiss.IndexFlatIP(dims))
        paragraphs_embeddings.save_faiss_index("embeddings", args.index_file_name)
    else:
        print(f"Faiss index already exists {args.index_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates Faiss Wikipedia index file")

    parser.add_argument(
        "--ctx_encoder_name",
        default="vblagoje/dpr-ctx_encoder-single-lfqa-base",
        help="Encoding model to use for passage encoding",
    )

    parser.add_argument(
        "--index_file_name",
        default="../data/kilt_dpr_wikipedia.faiss",
        help="Faiss index file with passage embeddings",
    )

    main_args, _ = parser.parse_known_args()
    create_faiss(main_args)
