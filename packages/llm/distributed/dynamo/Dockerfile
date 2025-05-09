#---
# name: dynamo
# group: distributed
# config: config.py
# depends: [cuda, rust, cmake, ninja, pytorch, torchvision, torchaudio, transformers, causalconv1d, triton, nixl, vllm, sglang, llvm:20]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/ai-dynamo/dynamo
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DYNAMO_VERSION \
    DYNAMO_VERSION_SPEC \
    IS_SBSA

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
