#---
# name: ktransformers
# group: llm
# config: config.py
# depends: [pytorch, torchvision, torchaudio, transformers, triton]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/ktransformers-project/ktransformers
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG KTRANSFORMERS_VERSION \
    CUDAARCHS \
    FORCE_BUILD=off

RUN apt-get update -y && apt-get install -y build-essential \
    cmake ninja-build

COPY build.sh install.sh /tmp/ktransformers/

RUN /tmp/ktransformers/install.sh || /tmp/ktransformers/build.sh
