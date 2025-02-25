#---
# name: vllm
# group: vlm
# config: config.py
# depends: [pytorch, torchvision, torchaudio, transformers, triton:3.1.0, bitsandbytes, xformers, flash-attention]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/vllm-project/vllm
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VLLM_VERSION \
    XGRAMMAR_VERSION \
    CUDAARCHS \
    FORCE_BUILD=off

RUN apt-get update -y && apt-get install -y libnuma-dev \
    libsndfile1 libsndfile1-dev libprotobuf-dev libsm6 libxext6 libgl1

COPY build.sh install.sh patches /tmp/vllm/

RUN /tmp/vllm/install.sh || /tmp/vllm/build.sh
