#---
# name: jvp-flash-attention
# group: attention
# config: config.py
# depends: [pytorch, triton, cutlass]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG JVP_FLASH_ATTENTION_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh patches /tmp/jvp-flash-attention/

RUN /tmp/jvp-flash-attention/install.sh || /tmp/jvp-flash-attention/build.sh

