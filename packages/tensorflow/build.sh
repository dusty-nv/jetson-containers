#!/usr/bin/env bash
# TensorFlow builder for Jetson (architecture: ARM64, CUDA support)
set -ex

echo "Building Tensorflow ${TENSORFLOW_VERSION}"

# Install LLVM/Clang 17 # Update to 18 when main will be ready. 
# Tensorflow will support llvm 18 and 19
./llvm.sh 17 all

echo "Building TensorFlow for Jetson"

# Clone the TensorFlow repository
git clone --branch "v${TENSORFLOW_VERSION}" --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow || \
git clone --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow 

wget https://github.com/bazelbuild/bazelisk/releases/download/v1.21.0/bazelisk-linux-arm64
chmod +x bazelisk-linux-arm64
mv bazelisk-linux-arm64 /usr/local/bin/bazel
cd /opt/tensorflow

# Set up environment variables for the configure script
export PYTHON_BIN_PATH="$(which python3)"
export PYTHON_LIB_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export TF_NEED_CUDA=1
export TF_CUDA_CLANG=1
# Set Clang path for CUDA
export CLANG_CUDA_COMPILER_PATH=/usr/local/llvm/bin/clang
# Set Clang path for CPU
export CLANG_COMPILER_PATH=/usr/local/llvm/bin/clang
export HERMETIC_CUDA_VERSION=12.6.0
export HERMETIC_CUDNN_VERSION=9.3.0 
export HERMETIC_CUDA_COMPUTE_CAPABILITIES=8.7
export HERMETIC_PYTHON_VERSION="${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"


# Build the TensorFlow pip package
bazel build //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow --config=cuda --config=cuda_wheel --config=nonccl --copt=-Wno-gnu-offsetof-extensions

# Upload the wheels to mirror
twine upload --verbose /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container
pip3 install --verbose --no-cache-dir /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl

# Verify the installation
python3 -c "import tensorflow as tf; print(tf.__version__)"