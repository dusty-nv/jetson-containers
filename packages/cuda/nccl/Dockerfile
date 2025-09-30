#---
# name: nccl
# group: cuda
# depends: [cuda]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NCCL_VERSION \
    CUDA_ARCH \
    IS_SBSA \
    FORCE_BUILD=off \
    DISTRO="ubuntu2004" \
    CUDA_ARCH \
    ENABLE_NCCL_DISTRIBUTED_JETSON=0 \
    TMP=/tmp/nccl

ENV ENABLE_NCCL_DISTRIBUTED_JETSON=${ENABLE_NCCL_DISTRIBUTED_JETSON}

RUN apt-get update -y && \
    apt-get install -y dkms && \
    apt-get install -y --no-install-recommends \
        build-essential \
        devscripts \
        debhelper \
        fakeroot \
        pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY install.sh build.sh patches $TMP/

RUN $TMP/install.sh || $TMP/build.sh
