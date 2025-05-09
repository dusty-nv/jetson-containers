#---
# name: dynamo
# group: llm
# config: config.py
# depends: [cuda, rust, cmake, ninja, pytorch, torchvision, torchaudio, torchao, transformers, triton]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/ai-dynamo/nixl
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NIXL_VERSION \
    CUDAARCHS

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/nixl/

RUN /tmp/nixl/install.sh || /tmp/nixl/build.sh
