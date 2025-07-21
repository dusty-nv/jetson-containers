#---
# name: cuda_cccl
# group: cuda
# config: config.py
# depends: [cuda, cudnn, python, cmake, ninja, cuda-python]
# requires: '>=35'
# test: test.py
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CCCL_VERSION \
    CCCL_VERSION_SPEC \
    TORCH_CUDA_ARCH_LIST \
    MAX_JOBS="$(nproc)" \
    IS_SBSA \
    FORCE_BUILD=off \
    TMP=/tmp/cuda_cccl

# 1) system deps, 2) newer GCC, 3) clean-up
RUN set -e; \
    apt-get update -y && \
    # cuda-cccl runtime & MPI
    apt-get install -y --no-install-recommends \
        cuda-cccl-* \
        libopenmpi-dev openmpi-bin \
    # modern compiler for <format>
        software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        gcc-13 g++-13 libstdc++-13-dev && \
    # make them the default
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    # tidy image
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

ENV CC=gcc CXX=g++

COPY build.sh install.sh ${TMP}/
RUN ${TMP}/install.sh || ${TMP}/build.sh || touch ${TMP}/.build.failed

# keep build layer if CCCL failed
RUN if [ -f ${TMP}/.build.failed ]; then \
        echo "cuda_cccl ${CCCL_VERSION} build failed!"; exit 1; \
    fi
