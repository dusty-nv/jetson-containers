#---
# name: dynamo
# group: dynamo
# config: config.py
# depends: [pytorch, torchvision, torchaudio, triton, transformers, mamba, nixl, llvm:21, vllm, sglang, flashinfer]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/ai-dynamo/dynamo
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DYNAMO_VERSION \
    DYNAMO_VERSION_SPEC \
    COMPUTE_CAPABILITIES \
    CUDA_COMPUTE_CAP \
    TORCH_CUDA_ARCH_LIST \
    SYSTEM_ARM

ENV COMPUTE_CAPABILITIES=${COMPUTE_CAPABILITIES} \
    CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

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
