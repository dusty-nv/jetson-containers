#---
# name: nvpl
# group: cuda
# depends: [cuda]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NVPL_VERSION

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

RUN git clone --recursive --depth 1 https://github.com/NVIDIA/NVPLSamples /opt/nvpl

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb /opt/nvpl && \
    dpkg -i /opt/nvpl/cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install nvpl \
    rm -rf /opt/nvpl/*.deb && \
    rm -rf /var/lib/apt/lists/*
