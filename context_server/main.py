import torch
from fastapi import FastAPI, Depends, status
from fastapi.responses import PlainTextResponse
from transformers import AutoTokenizer, AutoModel, DPRQuestionEncoder

from datasets import load_from_disk
import time
from typing import Dict

import jwt
from decouple import config
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

JWT_SECRET = config("secret")
JWT_ALGORITHM = config("algorithm")

app = FastAPI()
app.ready = False
columns = ['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text', 'anchors', 'categories',
           'wikidata_info', 'history']

min_snippet_length = 20
topk = 21
device = ("cuda" if torch.cuda.is_available() else "cpu")
model = DPRQuestionEncoder.from_pretrained("vblagoje/dpr-question_encoder-single-lfqa-wiki").to(device)
tokenizer = AutoTokenizer.from_pretrained("vblagoje/dpr-question_encoder-single-lfqa-wiki")
_ = model.eval()

index_file_name = "./data/kilt_wikipedia.faiss"

kilt_wikipedia_paragraphs = load_from_disk("./data/kilt_wiki_prepared")
# use paragraphs that are not simple fragments or very short sentences
kilt_wikipedia_paragraphs = kilt_wikipedia_paragraphs.filter(lambda x: x["end_character"] > 200)


class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(status_code=403, detail="Invalid token or expired token.")
            return credentials.credentials
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")

    def verify_jwt(self, jwtoken: str) -> bool:
        isTokenValid: bool = False

        try:
            payload = decodeJWT(jwtoken)
        except:
            payload = None
        if payload:
            isTokenValid = True
        return isTokenValid


def token_response(token: str):
    return {
        "access_token": token
    }


def signJWT(user_id: str) -> Dict[str, str]:
    payload = {
        "user_id": user_id,
        "expires": time.time() + 6000
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return token_response(token)


def decodeJWT(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded_token if decoded_token["expires"] >= time.time() else None
    except:
        return {}


def embed_questions_for_retrieval(questions):
    query = tokenizer(questions, max_length=128, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        q_reps = model(query["input_ids"].to(device), query["attention_mask"].to(device)).pooler_output
    return q_reps.cpu().numpy()

def query_index(question):
    question_embedding = embed_questions_for_retrieval([question])
    scores, wiki_passages = kilt_wikipedia_paragraphs.get_nearest_examples("embeddings", question_embedding, k=topk)
    columns = ['wikipedia_id', 'title', 'text', 'section', 'start_paragraph_id', 'end_paragraph_id',
               'start_character', 'end_character']
    retrieved_examples = []
    r = list(zip(wiki_passages[k] for k in columns))
    for i in range(topk):
        retrieved_examples.append({k: v for k, v in zip(columns, [r[j][0][i] for j in range(len(columns))])})
    return retrieved_examples


@app.on_event("startup")
def startup():
    kilt_wikipedia_paragraphs.load_faiss_index("embeddings", index_file_name, device=0)
    app.ready = True


@app.get("/healthz")
def healthz():
    if app.ready:
        return PlainTextResponse("ok")
    return PlainTextResponse("service unavailable", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.get("/find_context", dependencies=[Depends(JWTBearer())])
def find_context(question: str = None):
    return [res for res in query_index(question) if len(res["text"].split()) > min_snippet_length][:int(topk / 3)]

