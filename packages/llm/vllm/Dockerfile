#---
# name: vllm
# group: llm
# config: config.py
# depends: [transformers, bitsandbytes, triton, xformers, flash-attention, torchaudio, mamba, xgrammar, flashinfer, torch-memory-saver, mistral_common]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/vllm-project/vllm
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VLLM_VERSION \
    VLLM_BRANCH \
    CUDAARCHS \
    CUDA_SUFFIX \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh patches /tmp/vllm/

RUN /tmp/vllm/install.sh || /tmp/vllm/build.sh && \
    /tmp/transformers/install.sh
