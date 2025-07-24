#---
# name: unsloth
# group: llm
# config: config.py
# depends: [transformers, bitsandbytes, triton, xformers, flash-attention, torchaudio, mamba, xgrammar, flashinfer, minference, torch-memory-saver, mistral_common, ffmpeg]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/unsloth-project/unsloth
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG UNSLOTH_VERSION \
    UNSLOTH_BRANCH \
    CUDAARCHS \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/unsloth/

RUN /tmp/unsloth/install.sh || /tmp/unsloth/build.sh
RUN /tmp/transformers/install.sh
