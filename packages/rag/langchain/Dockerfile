#---
# name: langchain
# alias: langchain:main
# group: rag
# depends: [pytorch, llama_cpp]
# buildkit_device: nvidia.com/gpu=all
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN uv pip install langchain[llm] openai
