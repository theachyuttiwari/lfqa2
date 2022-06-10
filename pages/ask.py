import colorsys
import json
import re
import time

import nltk
import numpy as np
from nltk import tokenize

nltk.download('punkt')
from google.oauth2 import service_account
from google.cloud import texttospeech

from typing import Dict, Optional, List

import jwt
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer, util, CrossEncoder

JWT_SECRET = st.secrets["api_secret"]
JWT_ALGORITHM = st.secrets["api_algorithm"]
INFERENCE_TOKEN = st.secrets["api_inference"]
CONTEXT_API_URL = st.secrets["api_context"]
LFQA_API_URL = st.secrets["api_lfqa"]

headers = {"Authorization": f"Bearer {INFERENCE_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/vblagoje/bart_lfqa"
API_URL_TTS = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan"


def api_inference_lfqa(model_input: str):
    payload = {
        "inputs": model_input,
        "parameters": {
            "truncation": "longest_first",
            "min_length": st.session_state["min_length"],
            "max_length": st.session_state["max_length"],
            "do_sample": st.session_state["do_sample"],
            "early_stopping": st.session_state["early_stopping"],
            "num_beams": st.session_state["num_beams"],
            "temperature": st.session_state["temperature"],
            "top_k": None,
            "top_p": None,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": 1
        },
        "options": {
            "wait_for_model": True
        }
    }
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def inference_lfqa(model_input: str, header: dict):
    payload = {
        "model_input": model_input,
        "parameters": {
            "min_length": st.session_state["min_length"],
            "max_length": st.session_state["max_length"],
            "do_sample": st.session_state["do_sample"],
            "early_stopping": st.session_state["early_stopping"],
            "num_beams": st.session_state["num_beams"],
            "temperature": st.session_state["temperature"],
            "top_k": None,
            "top_p": None,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": 1
        }
    }
    data = json.dumps(payload)
    try:
        response = requests.request("POST", LFQA_API_URL, headers=header, data=data)
        if response.status_code == 200:
            json_response = response.content.decode("utf-8")
            result = json.loads(json_response)
        else:
            result = {"error": f"LFQA service unavailable, status code={response.status_code}"}
    except requests.exceptions.RequestException as e:
        result = {"error": e}
    return result


def invoke_lfqa(service_backend: str, model_input: str, header: Optional[dict]):
    if "HuggingFace" == service_backend:
        inference_response = api_inference_lfqa(model_input)
    else:
        inference_response = inference_lfqa(model_input, header)
    return inference_response


@st.cache(allow_output_mutation=True, show_spinner=False)
def hf_tts(text: str):
    payload = {
        "inputs": text,
        "parameters": {
            "vocoder_tag": "str_or_none(none)",
            "threshold": 0.5,
            "minlenratio": 0.0,
            "maxlenratio": 10.0,
            "use_att_constraint": False,
            "backward_window": 1,
            "forward_window": 3,
            "speed_control_alpha": 1.0,
            "noise_scale": 0.333,
            "noise_scale_dur": 0.333
        },
        "options": {
            "wait_for_model": True
        }
    }
    data = json.dumps(payload)
    response = requests.request("POST", API_URL_TTS, headers=headers, data=data)
    return response.content


@st.cache(allow_output_mutation=True, show_spinner=False)
def google_tts(text: str, private_key_id: str, private_key: str, client_email: str):
    config = {
        "private_key_id": private_key_id,
        "private_key": f"-----BEGIN PRIVATE KEY-----\n{private_key}\n-----END PRIVATE KEY-----\n",
        "client_email": client_email,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    credentials = service_account.Credentials.from_service_account_info(config)
    client = texttospeech.TextToSpeechClient(credentials=credentials)

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(language_code="en-US",
                                              ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response


def request_context_passages(question, header):
    try:
        response = requests.request("GET", CONTEXT_API_URL + question, headers=header)
        if response.status_code == 200:
            json_response = response.content.decode("utf-8")
            result = json.loads(json_response)
        else:
            result = {"error": f"Context passage service unavailable, status code={response.status_code}"}
    except requests.exceptions.RequestException as e:
        result = {"error": e}

    return result


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_sentence_transformer_encoding(sentences):
    model = get_sentence_transformer()
    return model.encode([sentence for sentence in sentences], convert_to_tensor=True)


def sign_jwt() -> Dict[str, str]:
    payload = {
        "expires": time.time() + 6000
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def extract_sentences_from_passages(passages):
    sentences = []
    for idx, node in enumerate(passages):
        sentences.extend(tokenize.sent_tokenize(node["text"]))
    return sentences


def similarity_color_picker(similarity: float):
    value = int(similarity * 75)
    rgb = colorsys.hsv_to_rgb(value / 300., 1.0, 1.0)
    return [round(255 * x) for x in rgb]


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % tuple(rgb)


def similiarity_to_hex(similarity: float):
    return rgb_to_hex(similarity_color_picker(similarity))


def rerank(question: str, passages: List[str], include_rank: int = 4) -> List[str]:
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    question_passage_combinations = [[question, p["text"]] for p in passages]

    # Compute the similarity scores for these combinations
    similarity_scores = ce.predict(question_passage_combinations)

    # Sort the scores in decreasing order
    sim_ranking_idx = np.flip(np.argsort(similarity_scores))
    return [passages[rank_idx] for rank_idx in sim_ranking_idx[:include_rank]]


def answer_to_context_similarity(generated_answer, context_passages, topk=3):
    context_sentences = extract_sentences_from_passages(context_passages)
    context_sentences_e = get_sentence_transformer_encoding(context_sentences)
    answer_sentences = tokenize.sent_tokenize(generated_answer)
    answer_sentences_e = get_sentence_transformer_encoding(answer_sentences)
    search_result = util.semantic_search(answer_sentences_e, context_sentences_e, top_k=topk)
    result = []
    for idx, r in enumerate(search_result):
        context = []
        for idx_c in range(topk):
            context.append({"source": context_sentences[r[idx_c]["corpus_id"]], "score": r[idx_c]["score"]})
        result.append({"answer": answer_sentences[idx], "context": context})
    return result


def post_process_answer(generated_answer):
    result = generated_answer
    # detect sentence boundaries regex pattern
    regex = r"([A-Z][a-z].*?[.:!?](?=$| [A-Z]))"
    answer_sentences = tokenize.sent_tokenize(generated_answer)
    # do we have truncated last sentence?
    if len(answer_sentences) > len(re.findall(regex, generated_answer)):
        drop_last_sentence = " ".join(s for s in answer_sentences[:-1])
        result = drop_last_sentence
    return result.strip()


def format_score(value: float, precision=2):
    return f"{value:.{precision}f}"


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_answer(question: str):
    if not question:
        return {}

    resp: Dict[str, str] = {}
    if question and len(question.split()) > 3:
        header = {"Authorization": f"Bearer {sign_jwt()}"}
        context_passages = request_context_passages(question, header)
        if "error" in context_passages:
            resp = context_passages
        else:
            context_passages = rerank(question, context_passages)
            conditioned_context = "<P> " + " <P> ".join([d["text"] for d in context_passages])
            model_input = f'question: {question} context: {conditioned_context}'

            inference_response = invoke_lfqa(st.session_state["api_lfqa_selector"], model_input, header)
            if "error" in inference_response:
                resp = inference_response
            else:
                resp["context_passages"] = context_passages
                resp["answer"] = post_process_answer(inference_response[0]["generated_text"])
    else:
        resp = {"error": f"A longer, more descriptive question will receive a better answer. '{question}' is too short."}
    return resp


def app():
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    footer = """
        <div class="footer-custom">
            Streamlit app - <a href="https://www.linkedin.com/in/danijel-petkovic-573309144/" target="_blank">Danijel Petkovic</a>  |   
            LFQA/DPR models - <a href="https://www.linkedin.com/in/blagojevicvladimir/" target="_blank">Vladimir Blagojevic</a>   |
            Guidance & Feedback - <a href="https://yjernite.github.io/" target="_blank">Yacine Jernite</a> |
            <a href="https://towardsdatascience.com/long-form-qa-beyond-eli5-an-updated-dataset-and-approach-319cb841aabb" target="_blank">Blog</a>
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

    st.title('Wikipedia Assistant')
    st.header('We are migrating to new backend infrastructure. ETA - 15.6.2022')

    #question = st.text_input(
    #    label='Ask Wikipedia an open-ended question below; for example, "Why do airplanes leave contrails in the sky?"')
    question = ""
    spinner = st.empty()
    if question !="":
        spinner.markdown(
            f"""
            <div class="loader-wrapper">
            <div class="loader">
            </div>
            <p>Generating answer for: <b>{question}</b></p>
            </div>
            <label class="loader-note">Answer generation may take up to 20 sec. Please stand by.</label>
        """,
            unsafe_allow_html=True,
        )

    question_response = get_answer(question)
    if question_response:
        if "error" in question_response:
            st.warning(question_response["error"])
        else:
            spinner.markdown(f"")
            generated_answer = question_response["answer"]
            context_passages = question_response["context_passages"]
            sentence_similarity = answer_to_context_similarity(generated_answer, context_passages, topk=3)
            sentences = "<div class='sentence-wrapper'>"
            for item in sentence_similarity:
                sentences += '<span>'
                score = item["context"][0]["score"]
                support_sentence = item["context"][0]["source"]
                sentences += "".join([                    
                        f'  {item["answer"]}',
                        f'<span style="background-color: #{similiarity_to_hex(score)}" class="tooltip">',
                            f'{format_score(score, precision=1)}',
                f'<span class="tooltiptext"><b>Wikipedia source</b><br><br> {support_sentence} <br><br>Similarity: {format_score(score)}</span>'
                ])
                sentences += '</span>'                
            sentences += '</span>'                
            st.markdown(sentences, unsafe_allow_html=True)

            with st.spinner("Generating audio..."):
                if st.session_state["tts"] == "HuggingFace":
                    audio_file = hf_tts(generated_answer)
                    with open("out.flac", "wb") as f:
                        f.write(audio_file)
                else:
                    audio_file = google_tts(generated_answer, st.secrets["private_key_id"],
                                            st.secrets["private_key"], st.secrets["client_email"])
                    with open("out.mp3", "wb") as f:
                        f.write(audio_file.audio_content)

                audio_file = "out.flac" if st.session_state["tts"] == "HuggingFace" else "out.mp3"
                st.audio(audio_file)

            st.markdown("""<hr></hr>""", unsafe_allow_html=True)

            model = get_sentence_transformer()

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Context")
            with col2:
                selection = st.selectbox(
                    label="", 
                    options=('Paragraphs', 'Sentences', 'Answer Similarity'), 
                    help="Context represents Wikipedia passages used to generate the answer")
            question_e = model.encode(question, convert_to_tensor=True)
            if selection == "Paragraphs":
                sentences = extract_sentences_from_passages(context_passages)
                context_e = get_sentence_transformer_encoding(sentences)
                scores = util.cos_sim(question_e.repeat(context_e.shape[0], 1), context_e)
                similarity_scores = scores[0].squeeze().tolist()
                for idx, node in enumerate(context_passages):
                    node["answer_similarity"] = "{0:.2f}".format(similarity_scores[idx])
                context_passages = sorted(context_passages, key=lambda x: x["answer_similarity"], reverse=True)
                st.json(context_passages)
            elif selection == "Sentences":
                sentences = extract_sentences_from_passages(context_passages)
                sentences_e = get_sentence_transformer_encoding(sentences)
                scores = util.cos_sim(question_e.repeat(sentences_e.shape[0], 1), sentences_e)
                sentence_similarity_scores = scores[0].squeeze().tolist()
                result = []
                for idx, sentence in enumerate(sentences):
                    result.append(
                        {"text": sentence, "answer_similarity": "{0:.2f}".format(sentence_similarity_scores[idx])})
                context_sentences = json.dumps(sorted(result, key=lambda x: x["answer_similarity"], reverse=True))
                st.json(context_sentences)
            else:
                st.json(sentence_similarity)
