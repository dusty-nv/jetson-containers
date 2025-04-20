#---
# name: dynamo
# group: llm
# config: config.py
# depends: [cuda, rust, transformers, causalconv1d, triton, sglang]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/ai-dynamo/dynamo
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DYNAMO_VERSION \
    CUDAARCHS

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libnuma-dev \
        libsndfile1 \
        libsndfile1-dev \
        libprotobuf-dev \
        libsm6 \
        libxext6 \
        libgl1 \
        protobuf-compiler \
        libucx0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/dynamo/

RUN /tmp/dynamo/install.sh || /tmp/dynamo/build.sh
