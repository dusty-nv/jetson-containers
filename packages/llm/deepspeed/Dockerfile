#---
# name: deepspeed
# group: llm
# config: config.py
# depends: [transformers, deepspeed-kernels, triton]
# requires: '>=35'
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEEPSPEED_VERSION \
    DEEPSPEED_BRANCH \
    DS_BUILD_OPS=1 \
    FORCE_BUILD="off"

COPY build.sh install.sh /tmp/deepspeed/

RUN /tmp/deepspeed/install.sh || /tmp/deepspeed/build.sh

