#---
# name: minference
# group: llm
# config: config.py
# depends: [ffmpeg, pytorch, torchvision, torchaudio, torchao, triton, flash-attention, mamba, huggingface_hub]
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

COPY build.sh install.sh $TMP/

# this retains the stage above for debugging on build failure
RUN if [ -f $TMP/.build.failed ]; then \
      echo "minference ${minference_VERSION} build failed!"; \
      exit 1; \
    fi
