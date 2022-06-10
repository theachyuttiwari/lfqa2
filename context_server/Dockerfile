FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04 
#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

WORKDIR /code

ENV HF_HOME=/code/cache

COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/app/main.py

COPY ./data/kilt_wiki_prepared/ /code/data/kilt_wiki_prepared

COPY ./data/kilt_wikipedia.faiss /code/data/kilt_wikipedia.faiss

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
