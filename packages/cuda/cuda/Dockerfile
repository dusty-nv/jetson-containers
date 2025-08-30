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
    CUDA_ARCH \
    CUDA_INSTALLED_VERSION \
    IS_SBSA \
    DISTRO="ubuntu2004"

COPY install.sh /tmp/cuda/install.sh
RUN /tmp/cuda/install.sh

ENV CUDA_HOME="/usr/local/cuda"
ENV NVCC_PATH="$CUDA_HOME/bin/nvcc"

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDAARCHS=${CUDA_ARCH_LIST} \
    CUDA_ARCHITECTURES=${CUDA_ARCH_LIST} \
    CUDA_INSTALLED_VERSION=${CUDA_INSTALLED_VERSION} \
    CUDA_HOME="/usr/local/cuda" \
    CUDNN_LIB_PATH="/usr/lib/aarch64-linux-gnu" \
    CUDNN_LIB_INCLUDE_PATH="/usr/include" \
    CMAKE_CUDA_COMPILER=${NVCC_PATH} \
    CUDA_NVCC_EXECUTABLE=${NVCC_PATH} \
    CUDACXX=${NVCC_PATH} \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CUDA_BIN_PATH="${CUDA_HOME}/bin" \
    CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    PATH="${PATH}:${CUDA_HOME}/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/compat:${CUDA_HOME}/lib64" \
    LDFLAGS="${LDFLAGS} -L/usr/local/cuda/lib64" \
    DEBIAN_FRONTEND=noninteractive


# === Fix for CCCL includes (CUDA 13+) ===
# CUDA 13 moved thrust/cub to a top-level 'cccl' directory.
# These paths are needed for compiling code that uses them.
# The stubs for -lcuda are NOT added to the runtime environment.
ENV CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:/usr/local/cuda/include/cccl" \
    C_INCLUDE_PATH="${C_INCLUDE_PATH}:/usr/local/cuda/include/cccl"
WORKDIR /
