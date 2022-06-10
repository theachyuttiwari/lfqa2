import torch
from transformers import AutoTokenizer, AutoModel

from datasets import load_dataset


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('vblagoje/retribert-base-uncased')
    model = AutoModel.from_pretrained('vblagoje/retribert-base-uncased').to(device)
    _ = model.eval()

    index_file_name = "./data/kilt_wikipedia.faiss"
    kilt_wikipedia = load_dataset("kilt_wikipedia", split="full")
    columns = ['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text', 'anchors', 'categories',
                              'wikidata_info', 'history']

    min_snippet_length = 20
    topk = 21

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

    kilt_wikipedia_paragraphs = kilt_wikipedia.map(articles_to_paragraphs, batched=True,
                                                   remove_columns=columns,
                                                   batch_size=256, cache_file_name=f"./wiki_kilt_paragraphs_full.arrow",
                                                   desc="Expanding wiki articles into paragraphs")

    # use paragraphs that are not simple fragments or very short sentences
    kilt_wikipedia_paragraphs = kilt_wikipedia_paragraphs.filter(lambda x: x["end_character"] > 250)
    kilt_wikipedia_paragraphs.load_faiss_index("embeddings", index_file_name, device=0)

    def embed_questions_for_retrieval(questions):
        query = tokenizer(questions, max_length=128, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            q_reps = model.embed_questions(query["input_ids"].to(device),
                                           query["attention_mask"].to(device)).cpu().type(torch.float)
        return q_reps.numpy()

    def query_index(question):
        question_embedding = embed_questions_for_retrieval([question])
        scores, wiki_passages = kilt_wikipedia_paragraphs.get_nearest_examples("embeddings", question_embedding, k=topk)
        columns = ['wikipedia_id', 'title', 'text', 'section', 'start_paragraph_id', 'end_paragraph_id', 'start_character','end_character']
        retrieved_examples = []
        r = list(zip(wiki_passages[k] for k in columns))
        for i in range(topk):
            retrieved_examples.append({k: v for k, v in zip(columns, [r[j][0][i] for j in range(len(columns))])})
        return retrieved_examples

    questions = ["What causes the contrails (cirrus aviaticus) behind jets at high altitude? ",
                 "Why does water heated to a room temeperature feel colder than the air around it?"]
    res_list = query_index(questions[0])
    res_list = [res for res in res_list if len(res["text"].split()) > min_snippet_length][:int(topk / 3)]
    for res in res_list:
        print("\n")
        print(res)


main()


