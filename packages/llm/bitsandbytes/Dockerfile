#---
# name: bitsandbytes
# group: llm
# config: config.py
# requires: '>=35'
# depends: [transformers, triton]
# test: test.py
# notes: https://github.com/bitsandbytes-foundation/bitsandbytes
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BITSANDBYTES_VERSION \
    FORCE_BUILD=off \
    BITSANDBYTES_REPO \
    BITSANDBYTES_BRANCH \
    CUDA_INSTALLED_VERSION \
    CUDA_MAKE_LIB \
    CUDAARCHS

COPY build.sh install.sh /tmp/bitsandbytes/

RUN /tmp/bitsandbytes/install.sh || /tmp/bitsandbytes/build.sh
