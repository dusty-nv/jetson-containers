#---
# name: dynamo
# group: dynamo
# config: config.py
# depends: [cuda, rust, cmake, ninja, pytorch, torchvision, torchaudio, transformers, mamba, triton, nixl, sglang, llvm:20]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/ai-dynamo/dynamo
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DYNAMO_VERSION \
    DYNAMO_VERSION_SPEC \
    IS_SBSA \
    VLLM_REF="0.8.4" \
    VLLM_PATCH="vllm_v0.8.4-dynamo-kv-disagg-patch.patch" \
    VLLM_PATCHED_PACKAGE_NAME="ai_dynamo_vllm" \
    VLLM_PATCHED_PACKAGE_VERSION="0.8.4.post1"

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
