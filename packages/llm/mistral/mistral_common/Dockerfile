#---
# name: mistral_common
# group: llm
# config: config.py
# depends: [transformers, bitsandbytes, triton, xformers, flash-attention, torchaudio, mamba, xgrammar, opencv]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/mistralai/mistral-common
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MISTRAL_COMMON_VERSION \
    MISTRAL_COMMON_BRANCH \
    CUDAARCHS \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/mistral_common/

RUN /tmp/mistral_common/install.sh || /tmp/mistral_common/build.sh
RUN /tmp/transformers/install.sh
