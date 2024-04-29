#---
# name: flash-attention
# group: llm
# depends: [pytorch]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLASH_ATTENTION_VERSION="2.5.7"

COPY build.sh install.sh patch.diff /tmp/flash-attention/

RUN /tmp/flash-attention/install.sh || /tmp/flash-attention/build.sh
    