#---
# name: minference
# group: llm
# config: config.py
# depends: [cuda, cuda-python, ffmpeg, pytorch, torchvision, torchaudio, torchao, triton, flash-attention, mamba, flexprefill, huggingface_hub, tilelang]
# requires: '>=36'
# test: test.sh
# notes: https://github.com/microsoft/MInference
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MINFERENCE_VERSION \
    MINFERENCE_VERSION_SPEC \
    MAX_JOBS="$(nproc)" \
    FORCE_BUILD=off

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libnuma-dev \
        libsndfile1 \
        libsndfile1-dev \
        libprotobuf-dev \
        libsm6 \
        libxext6 \
        libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh patches /tmp/minference/

RUN /tmp/minference/install.sh || /tmp/minference/build.sh
