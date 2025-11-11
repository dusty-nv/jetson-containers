#---
# name: sgl-kernel
# group: sglang
# config: config.py
# depends: [cuda, cmake, pytorch, triton, flash-attention, flashinfer, xgrammar, genai-bench, torchao]
# requires: '>=36'
# test: test.py
# notes: https://github.com/sgl-project/sglang/sgl-kernel
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SGL_KERNEL_VERSION \
    SGL_KERNEL_BRANCH \
    IS_SBSA \
    TORCH_CUDA_ARCH_LIST \
    CUDA_VERSION \
    FORCE_BUILD=off \
    TMP=/tmp/sglang

# Fixes docker build error "failed to create NVIDIA Container Runtime: requirements not met:
# unsatisfied condition: cuda>=12.9 (cuda=12.6): unknown"
# Due to Jetson having a different GPU architecture and NVML isn't fully supported.
# Disables the CUDA version check and version enforcement by the container runtime
ENV NVIDIA_REQUIRE_CUDA=""
ENV NVIDIA_DISABLE_REQUIRE="true"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libnuma-dev \
        libsndfile1 \
        libsndfile1-dev \
        libprotobuf-dev \
        libsm6 \
        libxext6 \
        libnuma-dev \
        libibverbs-dev \
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

RUN /tmp/transformers/install.sh
