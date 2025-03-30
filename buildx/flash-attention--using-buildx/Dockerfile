#---
# name: flash-attention
# group: llm
# config: config.py
# depends: [pytorch]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE=kairin/001:2025-03-25-0404
FROM ${BASE_IMAGE}

ARG FLASH_ATTENTION_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh patches /tmp/flash-attention/

RUN /tmp/flash-attention/install.sh || /tmp/flash-attention/build.sh
    
