import argparse
import json
import os

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DPRQuestionEncoder

from common import articles_to_paragraphs, kilt_wikipedia_columns
from common import kilt_wikipedia_paragraph_columns as columns


def eval_generate(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    question_tokenizer = AutoTokenizer.from_pretrained(args.question_encoder_name)
    question_model = DPRQuestionEncoder.from_pretrained(args.question_encoder_name).to(device)
    _ = question_model.eval()

    eli5_tokenizer = AutoTokenizer.from_pretrained('vblagoje/bart_eli5')
    eli5_model = AutoModelForSeq2SeqLM.from_pretrained('vblagoje/bart_eli5').to(device)
    _ = eli5_model.eval()

    min_snippet_length = 20
    topk = 21
    min_chars_per_passage = 200
    kilt_wikipedia = load_dataset("kilt_wikipedia", split="full")
    kilt_wikipedia_paragraphs = kilt_wikipedia.map(articles_to_paragraphs, batched=True,
                                                   remove_columns=kilt_wikipedia_columns,
                                                   batch_size=256,
                                                   cache_file_name=f"./data/wiki_kilt_paragraphs_full.arrow",
                                                   desc="Expanding wiki articles into paragraphs")

    # use paragraphs that are not simple fragments or very short sentences
    kilt_wikipedia_paragraphs = kilt_wikipedia_paragraphs.filter(
        lambda x: (x["end_character"] - x["start_character"]) > min_chars_per_passage)
    kilt_wikipedia_paragraphs.load_faiss_index("embeddings", args.index_file_name, device=0)

    def embed_questions_for_retrieval(questions):
        query = question_tokenizer(questions, max_length=128, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            q_reps = question_model(query["input_ids"].to(device),
                                    query["attention_mask"].to(device)).pooler_output
        return q_reps.cpu().numpy()

    def query_index(question):
        question_embedding = embed_questions_for_retrieval([question])
        scores, wiki_passages = kilt_wikipedia_paragraphs.get_nearest_examples("embeddings", question_embedding, k=topk)

        retrieved_examples = []
        r = list(zip(wiki_passages[k] for k in columns))
        for i in range(topk):
            retrieved_examples.append({k: v for k, v in zip(columns, [r[j][0][i] for j in range(len(columns))])})
        return retrieved_examples

    def create_kilt_datapoint(q_id, query, answer, res_list):
        # make a KILT data point
        # see https://github.com/facebookresearch/KILT#kilt-data-format

        provenance = [{
            "wikipedia_id": r["wikipedia_id"],  # *mandatory*
            "title": r["title"],
            "section": r["section"],
            "start_paragraph_id": r["start_paragraph_id"],
            "start_character": r["start_character"],
            "end_paragraph_id": r["end_paragraph_id"],
            "end_character": r["end_character"],
            "text": r["text"],
            "bleu_score": None,  # wrt original evidence
            "meta": None  # dataset/task specific
        } for r in res_list]

        output = [{"answer": answer, "provenance": provenance}]

        return {"id": q_id,
                "input": query,
                "output": output,  # each element is an answer or provenance (can have multiple of each)
                "meta": None  # dataset/task specific
                }

    kilt_output = []
    with open(args.kilt_input_file, "r") as f:
        kilt_items = [json.loads(x) for x in f.read().strip().split("\n")]
        progress_bar = tqdm(range(len(kilt_items)), desc="Creating KILT response document")
        for idx, item in enumerate(kilt_items):
            query = item["input"]
            res_list = query_index(query)

            res_list = [res for res in res_list if len(res["text"].split()) > min_snippet_length][:int(topk / 3)]
            documents = [res["text"] for res in res_list]
            conditioned_doc = "<P> " + " <P> ".join([d for d in documents])

            query_and_docs = "question: {} context: {}".format(query, conditioned_doc)

            model_input = eli5_tokenizer(query_and_docs, truncation=True, padding=True, return_tensors="pt")
            generated_answers_encoded = eli5_model.generate(input_ids=model_input["input_ids"].to(device),
                                                            attention_mask=model_input["attention_mask"].to(device),
                                                            min_length=50,
                                                            max_length=250,
                                                            do_sample=False,
                                                            early_stopping=True,
                                                            num_beams=8,
                                                            temperature=1.0,
                                                            top_k=None,
                                                            top_p=None,
                                                            no_repeat_ngram_size=3,
                                                            num_return_sequences=1)
            answer = eli5_tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)

            kilt_example = create_kilt_datapoint(item["id"], query, answer[0], res_list)
            kilt_output.append(kilt_example)
            progress_bar.update(1)

    with open(args.kilt_output_file, "w") as fp:
        for kilt_example in kilt_output:
            json.dump(kilt_example, fp)
            fp.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kilt_input_file', default="./eli5-dev-kilt.jsonl", type=str)
    parser.add_argument('--kilt_output_file', default="./eli5-predicted_retrieval.jsonl", type=str)
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

    args = parser.parse_args()

    assert os.path.isfile(args.kilt_input_file), f"Input file {args.kilt_input_file} couldn't be loaded"
    eval_generate(args)
