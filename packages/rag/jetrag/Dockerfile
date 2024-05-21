#---
# name: jetrag
# group: rag
# depends: [pytorch, ollama]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 install --no-cache-dir --verbose \
        llama-index-core \
        llama-index-readers-file \
        llama-index-llms-openai \
        llama-index-llms-ollama \
        llama-index-embeddings-openai \
        llama-index-embeddings-ollama \
        chromadb \
        pypdf \
        pytest \
        boto3 \
        jetson-stats \
        unstructured \
        streamlit \
        ollama

WORKDIR /opt/jetrag/

COPY app.py /opt/jetrag/app.py
COPY .streamlit/ /opt/jetrag/.streamlit
COPY static/ /opt/jetrag/static

CMD /start_ollama && /bin/bash
