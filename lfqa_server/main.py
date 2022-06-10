import torch
from fastapi import FastAPI, Depends, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import time
from typing import Dict, List, Optional

import jwt
from decouple import config
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

JWT_SECRET = config("secret")
JWT_ALGORITHM = config("algorithm")

app = FastAPI()
app.ready = False

device = ("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('vblagoje/bart_lfqa')
model = AutoModelForSeq2SeqLM.from_pretrained('vblagoje/bart_lfqa').to(device)
_ = model.eval()


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


class LFQAParameters(BaseModel):
    min_length: int = 50
    max_length: int = 250
    do_sample: bool = False
    early_stopping: bool = True
    num_beams: int = 8
    temperature: float = 1.0
    top_k: float = None
    top_p: float = None
    no_repeat_ngram_size: int = 3
    num_return_sequences: int = 1


class InferencePayload(BaseModel):
    model_input: str
    parameters: Optional[LFQAParameters] = LFQAParameters()


@app.on_event("startup")
def startup():
    app.ready = True


@app.get("/healthz")
def healthz():
    if app.ready:
        return PlainTextResponse("ok")
    return PlainTextResponse("service unavailable", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.post("/generate/", dependencies=[Depends(JWTBearer())])
def generate(context: InferencePayload):

    model_input = tokenizer(context.model_input, truncation=True, padding=True, return_tensors="pt")
    param = context.parameters
    generated_answers_encoded = model.generate(input_ids=model_input["input_ids"].to(device),
                                               attention_mask=model_input["attention_mask"].to(device),
                                               min_length=param.min_length,
                                               max_length=param.max_length,
                                               do_sample=param.do_sample,
                                               early_stopping=param.early_stopping,
                                               num_beams=param.num_beams,
                                               temperature=param.temperature,
                                               top_k=param.top_k,
                                               top_p=param.top_p,
                                               no_repeat_ngram_size=param.no_repeat_ngram_size,
                                               num_return_sequences=param.num_return_sequences)
    answers = tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    results = []
    for answer in answers:
        results.append({"generated_text": answer})
    return results
