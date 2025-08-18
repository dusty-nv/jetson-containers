#---
# name: flash-attention
# group: attention
# config: config.py
# depends: [pytorch, triton, cutlass]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLASH_ATTENTION_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh patches /tmp/flash-attention/

RUN /tmp/flash-attention/install.sh || /tmp/flash-attention/build.sh

