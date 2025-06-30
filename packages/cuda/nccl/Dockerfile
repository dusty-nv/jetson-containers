#---
# name: nccl
# group: cuda
# depends: [cuda]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NCCL_VERSION
ARG IS_SBSA

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

RUN git clone --recursive --depth 1 https://github.com/NVIDIA/ncclSamples /opt/nccl

# Instalar el keyring correcto según si IS_SBSA está definido
RUN mkdir -p /opt/nccl && \
    bash -c '\
    if [[ -z "${IS_SBSA:-}" || "${IS_SBSA}" == "1" || "${IS_SBSA,,}" == "true" ]]; then \
        echo "Using SBSA (ARM64) repo"; \
        URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb"; \
    else \
        echo "Using x86_64 repo"; \
        URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"; \
    fi && \
    wget "$URL" -O /opt/nccl/cuda-keyring.deb && \
    dpkg -i /opt/nccl/cuda-keyring.deb && \
    apt-get update && \
    apt-get -y install libnccl2 libnccl-dev && \
    rm -rf /opt/nccl/*.deb /var/lib/apt/lists/*'
