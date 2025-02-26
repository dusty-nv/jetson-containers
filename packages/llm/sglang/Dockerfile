#---
# name: sglang
# group: llm
# config: config.py
# depends: [pytorch, torchvision, torchaudio, transformers, triton:3.1.0, vllm, decord]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/sgl-project/sglang
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION \
    FLASHINFER_VERSION \
    CUDAARCHS \
    FORCE_BUILD=off

RUN apt-get update -y && apt-get install -y libnuma-dev \
    libsndfile1 libsndfile1-dev libprotobuf-dev libsm6 libxext6 libgl1 libavformat58 libavfilter7 && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/SGLANG/
RUN /tmp/SGLANG/install.sh || /tmp/SGLANG/build.sh

# Default command
CMD ["/bin/bash"]