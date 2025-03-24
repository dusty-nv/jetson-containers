#---
# name: bazel
# group: build
# depends: [build-essential]
# test: test.sh
#---


# Specify the base image using an ARG
ARG BASE_IMAGE=kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu
FROM ${BASE_IMAGE}

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    wget \
    python3.12 \
    python3-pip \
    libc6 \
    xfe \
    git

# Add NVIDIA package repository for Ubuntu 22.04
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-aarch64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update
    
# Install Bazelisk
RUN BAZELISK_RELEASE=$(wget -qO- https://api.github.com/repos/bazelbuild/bazelisk/releases/latest | grep -Po '"tag_name": "\K.*?(?=")') && \
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_RELEASE/bazelisk-linux-arm64" && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate $BAZELISK_URL -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Verify Bazelisk installation
RUN bazel --version

RUN apt-get update && apt-get upgrade -y

RUN pip3 check

RUN pip3 install pip
