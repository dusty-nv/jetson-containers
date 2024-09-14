#!/usr/bin/env bash
# TensorFlow builder for Jetson (architecture: ARM64, CUDA support)
set -ex

# Variables
TENSORFLOW_VERSION=${TENSORFLOW_BUILD_VERSION}  # Update this to match the desired TensorFlow version
CUDA_VERSION=12.2.0  # Update this to match your Jetson's CUDA version
CUDNN_VERSION=8.9.4.25  # Update this to match your Jetson's cuDNN version
CUDA_COMPUTE_CAPABILITIES="8.7"  # For Jetson Xavier NX (sm_72). Update according to your device.
LOCAL_CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
LOCAL_CUDNN_PATH="/usr/lib/aarch64-linux-gnu"

# Install LLVM/Clang 18
./llvm.sh 18 all

echo "Building TensorFlow for Jetson"

# Clone the TensorFlow repository
git clone --branch "v${TENSORFLOW_VERSION}" --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow || \
git clone --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow 

cd /opt/tensorflow

# Set up environment variables for the configure script
export TF_NEED_CUDA=1
export CUDA_TOOLKIT_PATH="${LOCAL_CUDA_PATH}"
export CUDNN_INSTALL_PATH="${LOCAL_CUDNN_PATH}"
export TF_CUDA_VERSION="${CUDA_VERSION}"
export TF_CUDNN_VERSION="${CUDNN_VERSION}"
export TF_CUDA_COMPUTE_CAPABILITIES="${CUDA_COMPUTE_CAPABILITIES}"
export TF_NEED_NCCL=0  # NCCL is typically not available on Jetson devices

# Run the configure script
./configure

# Build the TensorFlow pip package
bazel build --config=opt --config=cuda --config=cuda_wheel \
    //tensorflow/tools/pip_package:wheel \
    --repo_env=WHEEL_NAME=tensorflow

# Upload the wheels to mirror
twine upload --verbose /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container
pip3 install --verbose --no-cache-dir /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl

# Verify the installation
python3 -c "import tensorflow as tf; print(tf.__version__)"