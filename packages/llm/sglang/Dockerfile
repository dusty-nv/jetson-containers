#---
# name: sglang
# group: llm
# config: config.py
# depends: [vllm, flashinfer, torchao, cuda-python, ffmpeg]
# requires: '>=36'
# test: test.sh
# notes: https://github.com/sgl-project/sglang
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION \
    SGLANG_VERSION_SPEC \
    SGL_KERNEL_ENABLE_BF16=1 \
    SGL_KERNEL_ENABLE_FP8=1 \
    SGL_KERNEL_ENABLE_FP4=0 \
    MAX_JOBS="$(nproc)" \
    FORCE_BUILD=off \
    TMP=/tmp/sglang

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
RUN $TMP/install.sh || $TMP/build.sh || touch $TMP/.build.failed

# this retains the stage above for debugging on build failure
RUN if [ -f $TMP/.build.failed ]; then \
      echo "SGLANG ${SGLANG_VERSION} build failed!"; \
      exit 1; \
    fi
