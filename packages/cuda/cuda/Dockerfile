#---
# name: cuda
# group: cuda
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_URL
ARG CUDA_DEB
ARG CUDA_PACKAGES

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            wget \
		  git \
		  binutils \
		  xz-utils \
		  ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN echo "Downloading ${CUDA_DEB}" && \
    mkdir /tmp/cuda && cd /tmp/cuda && \
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

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /