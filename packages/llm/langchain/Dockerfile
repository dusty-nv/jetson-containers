#---
# name: langchain
# group: llm
# depends: [pytorch, llama_cpp]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 install --no-cache-dir --verbose langchain[llm] openai
