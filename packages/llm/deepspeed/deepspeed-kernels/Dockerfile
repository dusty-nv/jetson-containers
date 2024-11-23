#---
# name: deepspeed-kernel
# group: llm
# config: config.py
# depends: [cuda, python, cmake]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEEPSPEED_KERNELS_VERSION \
    DEEPSPEED_KERNELS_BRANCH \
    FORCE_BUILD="off"

ADD https://api.github.com/repos/microsoft/DeepSpeed-Kernels/git/refs/heads/${DEEPSPEED_KERNELS_BRANCH} /tmp/deepspeed_kernels_version.json

COPY build.sh install.sh /tmp/deepspeed-kernels/

RUN /tmp/deepspeed-kernels/install.sh || /tmp/deepspeed-kernels/build.sh

