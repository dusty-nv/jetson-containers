#---
# name: hymba
# group: llm
# depends: [pytorch, transformers, xformers, flash-attention, mamba, triton, huggingface_hub]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NVlabs/hymba
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN pip3 install --no-cache-dir einops
# Clona el repositorio de Diffusers
RUN git clone https://github.com/NVlabs/hymba.git && \
    cd hymba && \
    python3 chat.py
