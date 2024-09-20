#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${TENSORFLOW_GRAPHICS_VERSION} --depth=1 --recursive https://github.com/tensorflow/graphics /opt/tensorflow_graphics || \
git clone --depth=1 --recursive https://github.com/tensorflow/graphics /opt/tensorflow_graphics

cd /opt/tensorflow_graphics 

export HERMETIC_PYTHON_VERSION="${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
export PYTHON_BIN_PATH="$(which python3)"
export PYTHON_LIB_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export TF_NEED_CUDA=1
export TF_CUDA_CLANG=1
export CLANG_CUDA_COMPILER_PATH="/usr/lib/llvm-17/bin/clang"
export HERMETIC_CUDA_VERSION=12.6.0
export HERMETIC_CUDNN_VERSION=9.3.0 
export HERMETIC_CUDA_COMPUTE_CAPABILITIES=8.7

pip3 wheel --no-build-isolation --wheel-dir=/opt/tensorflow_graphics/wheels .
pip3 install --no-cache-dir --verbose /opt/tensorflow_graphics/wheels/tensorflow-graphics*.whl

cd /opt/tensorflow_graphics
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/tensorflow_graphics/wheels/tensorflow-graphics*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
