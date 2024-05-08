#---
# name: flash-attention
# group: llm
# config: config.py
# depends: [pytorch]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLASH_ATTENTION_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/flash-attention/
COPY patches/${FLASH_ATTENTION_VERSION}.diff /tmp/flash-attention/patch.diff

RUN /tmp/flash-attention/install.sh || /tmp/flash-attention/build.sh
    
