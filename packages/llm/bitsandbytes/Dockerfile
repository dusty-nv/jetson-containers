#---
# name: bitsandbytes
# group: llm
# config: config.py
# requires: '>=35'
# depends: [transformers, openai-triton]
# test: test.py
# notes: fork of https://github.com/TimDettmers/bitsandbytes for Jetson
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BITSANDBYTES_VERSION \
    FORCE_BUILD=off \
    BITSANDBYTES_REPO \
    BITSANDBYTES_BRANCH \
    CUDA_INSTALLED_VERSION \
    CUDA_MAKE_LIB 

COPY build.sh install.sh /tmp/bitsandbytes/

RUN /tmp/bitsandbytes/install.sh || /tmp/bitsandbytes/build.sh
