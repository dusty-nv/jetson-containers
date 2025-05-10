#---
# name: dynamo_vllm
# group: dynamo
# config: config.py
# depends: [transformers, bitsandbytes, triton, xformers, flash-attention, torchaudio, mamba, xgrammar, flashinfer:0.2.2.post1, minference]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/vllm-project/vllm
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDAARCHS \
    VLLM_REF="0.8.4" \
    VLLM_PATCH="vllm_v0.8.4-dynamo-kv-disagg-patch.patch" \
    VLLM_PATCHED_PACKAGE_NAME="ai_dynamo_vllm" \
    VLLM_PATCHED_PACKAGE_VERSION="0.8.4.post1" \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh patches /tmp/dynamo_vllm/

RUN /tmp/dynamo_vllm/install.sh || /tmp/dynamo_vllm/build.sh
RUN /tmp/transformers/install.sh
