#!/usr/bin/env bash
# JAX builder for Jetson AGX (architecture: ARM64, CUDA support)
set -ex

echo "Building JAX for Jetson AGX Orin"

# Install prerequisites
apt-get update
apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        gfortran \
        libomp-dev \
        python3-dev \
        clang \
        python3-pip \
        g++

# Clean up package lists
rm -rf /var/lib/apt/lists/*
apt-get clean

# Clone JAX repository
git clone --branch "jax-v${JAX_BUILD_VERSION}" --depth=1 --recursive https://github.com/google/jax /opt/jax || \
git clone --depth=1 --recursive https://github.com/google/jax /opt/jax
cd /opt/jax

# Build jaxlib from source with detected versions
#python3 build/build.py  --enable_cuda --cuda_version 12.2 --cudnn_version 9 --cuda_compute_capabilities sm_87 --cuda_path /usr/local/cuda-12.2 --cudnn_path /usr/lib/aarch64-linux-gnu 

python3 build/build.py --enable_cuda --cuda_compute_capabilities=sm_87 --bazel_options=--repo_env=LOCAL_CUDA_PATH="/usr/local/cuda-12.2" --bazel_options=--repo_env=LOCAL_CUDNN_PATH="/usr/lib/aarch64-linux-gnu"

# --bazel_options=--repo_env=LOCAL_NCCL_PATH="/foo/bar/nvidia/nccl"
# Install the built JAX package
pip3 install -e .

# Validate JAX installation
python3 -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"CUDA devices: {jax.devices()}")'

# Clean up after build
cd /
rm -rf /opt/jax

echo "JAX build and installation complete."
