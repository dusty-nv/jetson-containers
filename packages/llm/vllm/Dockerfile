#---
# name: vllm
# group: llm
# config: config.py
# depends: [transformers, bitsandbytes, triton, xformers, flash-attention, torchaudio, mamba, xgrammar, flashinfer:0.2.2.post1, minference]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/vllm-project/vllm
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VLLM_VERSION \
    VLLM_BRANCH \
    XGRAMMAR_VERSION \
    CUDAARCHS \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh patches /tmp/vllm/

RUN /tmp/vllm/install.sh || /tmp/vllm/build.sh
RUN /tmp/transformers/install.sh
