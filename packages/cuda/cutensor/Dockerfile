#---
# name: cutensor
# group: cuda
# depends: [cuda]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUTENSOR_VERSION \
    CUDA_ARCH \
    IS_SBSA \
    FORCE_BUILD=off \
    CUDA_VERSION_MAJOR \
    DISTRO="ubuntu2004" \
    TMP=/tmp/cutensor

RUN apt-get update && \
    apt-get install -y dkms

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    devscripts \
    debhelper \
    fakeroot \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY install.sh /tmp/cutensor/install.sh
RUN /tmp/cutensor/install.sh
