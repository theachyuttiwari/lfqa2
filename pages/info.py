import streamlit as st


def app():
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    footer = """
           <div class="footer-custom">
               Streamlit app - <a href="https://www.linkedin.com/in/danijel-petkovic-573309144/" target="_blank">Danijel Petkovic</a>  |   
               LFQA/DPR models - <a href="https://www.linkedin.com/in/blagojevicvladimir/" target="_blank">Vladimir Blagojevic</a>   |
               Guidance & Feedback - <a href="https://yjernite.github.io/" target="_blank">Yacine Jernite</a>
           </div>
       """
    st.markdown(footer, unsafe_allow_html=True)

    st.subheader("Intro")
    intro = """
    <div class="text">
    Wikipedia Assistant is an example of a task usually referred to as the Long-Form Question Answering (LFQA). 
    These systems function by querying large document stores for relevant information and subsequently using 
    the retrieved documents to generate accurate, multi-sentence answers. The documents related to a given 
    query, colloquially called context passages, are not used merely as source tokens for extracted answers, 
    but instead provide a larger context for the synthesis of original, abstractive long-form answers. 
    LFQA systems usually consist of three components:
        <ul>
        <li>A document store including content passages for a variety of topics</li>
        <li>Encoder models to encode documents/questions such that it is possible to query the document store</li> 
        <li>A Seq2Seq language model capable of generating paragraph-long answers when given a question and 
        context passages retrieved from the document store</li>
        </ul> 
    </div>
    <br>
    """
    st.markdown(intro, unsafe_allow_html=True)
    st.image("lfqa.png", caption="LFQA Architecture")
    st.subheader("UI/UX")
    st.write("Each sentence in the generated answer ends with a coloured tooltip; the colour ranges from red to green. "
             "The tooltip contains a value representing answer sentence similarity to a specific sentence in the "
             "Wikipedia context passages retrieved.  Mouseover on the tooltip will show the sentence from the "
             "Wikipedia context passage. If a sentence similarity is 1.0, the seq2seq model extracted and "
             "copied the sentence verbatim from Wikipedia context passages. Lower values of sentence "
             "similarity indicate the seq2seq model is struggling to generate a relevant sentence for the question "
             "asked.")
    st.image("wikipedia_answer.png", caption="Answer with similarity tooltips")
    st.write("Below the generated answer are question-related Wikipedia context paragraphs (passages). One can view "
             "these passages in a raw format retrieved using the 'Paragraphs' select menu option. The 'Sentences' menu "
             "option shows the same paragraphs but on a sentence level. Finally, the 'Answer Similarity' menu option "
             "shows the most similar three sentences from context paragraphs to each sentence in the generated answer.")
    st.image("wikipedia_context.png", caption="Context paragraphs (passages)")

    tts = """    
    <div class="text">
    Wikipedia Assistant converts the text-based answer to speech via either Google text-to-speech engine or 
    <a href="https://github.com/espnet" target=_blank">Espnet model</a> hosted on 
    <a href="https://huggingface.co/espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan" target=_blank"> 
    HuggingFace hub</a> 
    <br>
    <br>
    """
    st.markdown(tts, unsafe_allow_html=True)

    st.subheader("Tips")
    tips = """
    <div class="text">
    LFQA task is far from solved. Wikipedia Assistant will sometimes generate an answer unrelated to a question asked, 
    even downright wrong. However, if the question is elaborate and more specific, there is a decent chance of 
    getting a legible answer. LFQA systems are targeting ELI5 non-factoid type of questions. A general guideline 
    is - questions starting with why, what, and how are better suited than where and who questions. Be elaborate. 
    <br><br>
    For example, to ask a history-based question, Wikipedia Assistant is better suited to answer the question: 
    "What was the objective of the German commando raid on Drvar in Bosnia during the Second World War?" than 
    "Why did Germans raid Drvar?". A precise science question like "Why do airplane jet engines leave contrails 
    in the sky?" has a good chance of getting a decent answer. Detailed and precise questions are more likely to 
    match the right half a dozen relevant passages in a 20+ GB Wikipedia dump to construct a good answer.
    </div>
    <br>  
    """
    st.markdown(tips, unsafe_allow_html=True)
    st.subheader("Technical details")
    techinical_intro = """    
    <div class="text technical-details-info">
        A question asked will be encoded with an <a href="https://huggingface.co/vblagoje/dpr-question_encoder-single-lfqa-wiki" target=_blank">encoder</a> 
        and sent to a server to find the most relevant Wikipedia passages. The Wikipedia <a href="https://huggingface.co/datasets/kilt_wikipedia" target=_blank">passages</a> 
        were previously encoded using a passage <a href="https://huggingface.co/vblagoje/dpr-ctx_encoder-single-lfqa-wiki" target=_blank">encoder</a> and 
        stored in the <a href="https://github.com/facebookresearch/faiss" target=_blank">Faiss</a> index. The question matching passages (a.k.a context passages) are retrieved from the Faiss 
        index and passed to a BART-based seq2seq <a href="https://huggingface.co/vblagoje/bart_lfqa" target=_blank">model</a> to
        synthesize an original answer to the question. 
        
    </div>
    """
    st.markdown(techinical_intro, unsafe_allow_html=True)

