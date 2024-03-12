#---
# name: cuda
# group: cuda
# config: config.py
# depends: [build-essential]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_URL
ARG CUDA_DEB
ARG CUDA_PACKAGES
ARG CUDA_ARCH_LIST

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            wget \
		  git \
		  binutils \
		  xz-utils \
		  ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean && \
    \
    echo "Downloading ${CUDA_DEB}" && \
    mkdir /tmp/cuda && cd /tmp/cuda && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-ubuntu2004.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget --quiet --show-progress --progress=bar:force:noscroll ${CUDA_URL} && \
    dpkg -i *.deb && \
    cp /var/cuda-tegra-repo-*/cuda-tegra-*-keyring.gpg /usr/share/keyrings/ && \
    ar x /var/cuda-tegra-repo-*/cuda-compat-*.deb && \
    tar xvf data.tar.xz -C / && \
    apt-get update && \
    apt-get install -y --no-install-recommends ${CUDA_PACKAGES} && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    dpkg --list | grep cuda && \
    dpkg -P ${CUDA_DEB} && \
    rm -rf /tmp/cuda

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDAARCHS=${CUDA_ARCH_LIST} \
    CUDA_ARCHITECTURES=${CUDA_ARCH_LIST} \
    CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
