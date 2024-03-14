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
    CUDA_ARCH_LIST \
    DISTRO="ubuntu2004" \
    DEBIAN_FRONTEND=noninteractive

RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    \
    && echo "Downloading ${CUDA_DEB}" \
    && mkdir -p /tmp/cuda \
    && cd /tmp/cuda \
    && wget --quiet --show-progress --progress=bar:force:noscroll https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/arm64/cuda-${DISTRO}.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget --quiet --show-progress --progress=bar:force:noscroll ${CUDA_URL} \
    && dpkg -i *.deb \
    && cp /var/cuda-tegra-repo-ubuntu*-local/cuda-tegra-*-keyring.gpg /usr/share/keyrings/ \
    && ar x /var/cuda-tegra-repo-ubuntu*-local/cuda-compat-*.deb \
    && tar xvf data.tar.xz -C / \
    \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${CUDA_PACKAGES} \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    \
    && dpkg --list | grep cuda \
    && dpkg -P ${CUDA_DEB} \
    && rm -rf /tmp/cuda

ENV CUDA_HOME="/usr/local/cuda"
ENV NVCC_PATH="$CUDA_HOME/bin/nvcc"
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDAARCHS=${CUDA_ARCH_LIST} \
    CUDA_ARCHITECTURES=${CUDA_ARCH_LIST} \
    CUDNN_LIB_PATH="/usr/lib/aarch64-linux-gnu" \
    CUDNN_LIB_INCLUDE_PATH="/usr/include" \
    CMAKE_CUDA_COMPILER=${NVCC_PATH} \
    CUDA_NVCC_EXECUTABLE=${NVCC_PATH} \
    CUDACXX=${NVCC_PATH} \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CUDA_BIN_PATH="${CUDA_HOME}/bin" \
    CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    PATH="$CUDA_HOME/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/compat:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /
