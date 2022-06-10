import streamlit as st
from multipage import MultiPage
from pages import ask, settings, info


def init_session_key_value(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


lfqa_api = "HuggingFace" if "api_lfqa_selector" not in st.secrets else st.secrets["api_lfqa_selector"]
session_values = {"api_lfqa_selector": lfqa_api,
                  "tts": "Google",
                  "min_length": 64,
                  "max_length": 256,
                  "do_sample": False,
                  "early_stopping": True,
                  "num_beams": 8,
                  "temperature": 1.0,
                  "top_k": None,
                  "top_p": None,
                  "no_repeat_ngram_size": 3,
                  "num_return_sequences": 1}

for k, v in session_values.items():
    init_session_key_value(k, v)

app = MultiPage()
st.set_page_config(
    page_title="Wikipedia Assistant",
    initial_sidebar_state="expanded",
)
# Add all your application here
app.add_page("Home", "house", ask.app)
app.add_page("Settings", "gear", settings.app)
app.add_page("Info", "info", info.app)

# The main app
app.run()
