#---
# name: llama-index
# alias: llama-index:main
# group: llm
# depends: [pytorch, ollama]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 install --no-cache-dir --verbose llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface

WORKDIR /opt/llama-index

CMD ["/bin/bash"]