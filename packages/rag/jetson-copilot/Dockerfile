#---
# name: jetson-copilot
# alias: jetrag
# group: rag
# depends: [pytorch, ollama, transformers]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN uv pip install \
        llama-index-core \
        llama-index-readers-file \
        llama-index-llms-openai \
        llama-index-llms-ollama \
        llama-index-embeddings-openai \
        llama-index-embeddings-ollama \
        llama-index-readers-web \
        openpyxl python-pptx Pillow docx2txt\
        chromadb \
        pypdf \
        pytest \
        boto3 \
        jetson-stats \
        unstructured \
        streamlit \
        ollama

WORKDIR /opt/jetson-copilot/

COPY app.py /opt/jetson-copilot/app.py
COPY .streamlit/ /opt/jetson-copilot/.streamlit
COPY static/ /opt/jetson-copilot/static

CMD /start_ollama && /bin/bash
