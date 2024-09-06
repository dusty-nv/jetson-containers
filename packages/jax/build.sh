#!/usr/bin/env bash
# JAX builder for Jetson AGX Orin (architecture: ARM64, CUDA support)
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
        bazel \
        python3-pip

# Clean up package lists
rm -rf /var/lib/apt/lists/*
apt-get clean

# Install Python dependencies
pip3 install --no-cache-dir numpy scipy scikit-build ninja

# Detect Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

# Detect CUDA version
CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
echo "Detected CUDA version: $CUDA_VERSION"

# Detect cuDNN version
CUDNN_VERSION=$(cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 | awk '{print $3}' | xargs | sed 's/ /./g')
echo "Detected cuDNN version: $CUDNN_VERSION"

# Clone JAX repository
git clone --branch "main" --depth=1 https://github.com/google/jax /opt/jax
cd /opt/jax

# Clone and install jaxlib, which is required for building JAX
pip3 install --no-cache-dir jaxlib

# Build jaxlib from source with detected versions
python3 build/build.py --python_version=$PYTHON_VERSION --enable_cuda --cuda_version=$CUDA_VERSION --cudnn_version=$CUDNN_VERSION

# Install the built JAX package
pip3 install -e .

# Validate JAX installation
python3 -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"CUDA devices: {jax.devices()}")'

# Clean up after build
cd /
rm -rf /opt/jax

echo "JAX build and installation complete."
