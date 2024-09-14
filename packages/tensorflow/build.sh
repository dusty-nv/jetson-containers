#!/usr/bin/env bash
# tensorflow builder for Jetson (architecture: ARM64, CUDA support)
set -ex

# Install LLVM/clang dev packages
./llvm.sh 18 all

echo "Building Tensorflow for Jetson"

# Clone tensorflow repository
git clone --branch "v${TENSORFLOW_BUILD_VERSION}" --depth=1 --recursive https://github.com/tensorflow/tensorflow /opt/tensorflow || \
git clone --depth=1 --recursive https://github.com/tensorflow/tensorflow /opt/tensorflow

cd /opt/tesnorflow

# python3 configure.py 
# Build tensorflow from source with detected versions
BUILD_FLAGS='--enable_cuda --enable_nccl=False '
BUILD_FLAGS+='--cuda_compute_capabilities="sm_87" '
BUILD_FLAGS+='--cuda_version=12.6.0 --cudnn_version=9.4.0 '
BUILD_FLAGS+='--bazel_options=--repo_env=LOCAL_CUDA_PATH="/usr/local/cuda-12.6" '
BUILD_FLAGS+='--bazel_options=--repo_env=LOCAL_CUDNN_PATH="/opt/nvidia/cudnn/" '
BUILD_FLAGS+='--output_path=/opt/wheels '
    
bazel build //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow --config=cuda --config=cuda_wheel

# Upload the wheels to mirror
twine upload --verbose /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container
pip3 install --verbose --no-cache-dir /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl