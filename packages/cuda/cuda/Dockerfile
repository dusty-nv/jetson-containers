#---
# name: cuda
# group: cuda
# config: config.py
# depends: [build-essential]
# test: test.sh
#---
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG CUDA_URL \
    CUDA_DEB \
    CUDA_PACKAGES \
    CUDA_TEMP_DIR="/tmp/cuda" \
    UBUNTU_VERSION="2004" \
    DEBIAN_FRONTEND=noninteractive

RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
        git \
        binutils \
        xz-utils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    \
    && echo "Downloading ${CUDA_DEB}" \
    && mkdir -p ${CUDA_TEMP_DIR} \
    && cd ${CUDA_TEMP_DIR} \
    && wget --quiet --show-progress --progress=bar:force:noscroll https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/arm64/cuda-ubuntu${UBUNTU_VERSION}.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget --quiet --show-progress --progress=bar:force:noscroll ${CUDA_URL} \
    && dpkg -i *.deb \
    && cp /var/cuda-tegra-repo-ubuntu*-local/cuda-tegra-*-keyring.gpg /usr/share/keyrings/ \
    && ar x /var/cuda-tegra-repo-ubuntu*-local/cuda-compat-*.deb \
    && tar xvf data.tar.xz -C / \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${CUDA_PACKAGES} \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && dpkg --list | grep cuda \
    && dpkg -P ${CUDA_DEB} \
    && rm -rf ${CUDA_TEMP_DIR}

ENV CUDA_HOME="/usr/local/cuda"
ENV NVCC_PATH="$CUDA_HOME/bin/nvcc"
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDNN_LIB_PATH="/usr/lib/aarch64-linux-gnu" \
	CUDNN_LIB_INCLUDE_PATH="/usr/include" \
    CMAKE_CUDA_COMPILER=$NVCC_PATH \
	CUDA_NVCC_EXECUTABLE=$NVCC_PATH \
	CUDACXX=$NVCC_PATH \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
	CUDA_HOST_COMPILER=$GCC_PATH \
	CUDA_BIN_PATH="$CUDA_HOME/bin" \
	CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    PATH="$CUDA_HOME/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/compat:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /
