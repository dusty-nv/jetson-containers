#---
# name: llama-index
# alias: llama-index:main
# group: rag
# depends: [pytorch]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN uv pip install \
        llama-index-core \
        llama-index-readers-file \
        llama-index-llms-ollama \
        llama-index-embeddings-huggingface

CMD /bin/bash
