import re

import torch

kilt_wikipedia_columns = ['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text', 'anchors', 'categories',
                          'wikidata_info', 'history']

kilt_wikipedia_paragraph_columns = ['wikipedia_id', 'start_paragraph_id', 'start_character', 'end_paragraph_id',
                                    'end_character', 'title', 'section', 'text']


def clean_question(text):
    result = cleanup_references(text)
    result = result.replace("\n", " ")
    result = re.sub(r"\s\s+", " ", result)
    result = result.replace("[deleted]", "")
    return result.lower().strip()


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
    return trim(result.strip())


def trim(text, word_count: int = 100):
    return " ".join(text.split(" ")[:word_count])


def articles_to_paragraphs(examples):
    ids, titles, sections, texts, start_ps, end_ps, start_cs, end_cs = [], [], [], [], [], [], [], []
    for bidx, example in enumerate(examples["text"]):
        last_section = ""
        for idx, p in enumerate(example["paragraph"]):
            if "Section::::" in p:
                last_section = p
            ids.append(examples["wikipedia_id"][bidx])
            titles.append(examples["wikipedia_title"][bidx])
            sections.append(last_section)
            texts.append(p)
            start_ps.append(idx)
            end_ps.append(idx)
            start_cs.append(0)
            end_cs.append(len(p))

    return {"wikipedia_id": ids, "title": titles,
            "section": sections, "text": texts,
            "start_paragraph_id": start_ps, "end_paragraph_id": end_ps,
            "start_character": start_cs,
            "end_character": end_cs
            }


def create_kilt_datapoint(eli5_example, columns, wiki_passages, min_length=20, topk=7):
    res_list = [dict([(k, p[k]) for k in columns]) for p in wiki_passages]
    res_list = [res for res in res_list if len(res["text"].split()) > min_length][:topk]

    # make a KILT data point
    # see https://github.com/facebookresearch/KILT#kilt-data-format
    output = []
    for a in eli5_example["answers"]["text"]:
        output.append({"answer": a})

    output.append({"provenance": [
        # evidence set for the answer from the KILT ks
        {
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
        } for r in res_list
    ]})
    return {"id": eli5_example["q_id"],
            "input": eli5_example["title"],
            "output": output,  # each element is an answer or provenance (can have multiple of each)
            "meta": None  # dataset/task specific
            }


def embed_questions(question_model, question_tokenizer, questions, max_length=128, device="cuda:0"):
    query = question_tokenizer(questions, max_length=max_length, padding="max_length", truncation=True,
                               return_tensors="pt")
    with torch.no_grad():
        q_reps = question_model(query["input_ids"].to(device),
                                query["attention_mask"].to(device)).pooler_output
    return q_reps.cpu().numpy()


def embed_passages(ctx_model, ctx_tokenizer, passages, max_length=128, device="cuda:0"):
    p = ctx_tokenizer(passages["text"], max_length=max_length, padding="max_length",
                      truncation=True, return_tensors="pt")
    with torch.no_grad():
        a_reps = ctx_model(p["input_ids"].to(device),
                           p["attention_mask"].to(device)).pooler_output
    return {"embeddings": a_reps.cpu().numpy()}
