#!/usr/bin/env bash

set -e

git clone --recursive --branch=v${TENSORFLOW_VERSION} --depth=1 https://github.com/tensorflow/text /opt/tensorflow-text  || \
git clone --depth=1 --recursive https://github.com/tensorflow/text /opt/tensorflow-text

cd /opt/tensorflow-text/

export HERMETIC_PYTHON_VERSION="${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
export PYTHON_BIN_PATH="$(which python3)"
export PYTHON_LIB_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export TF_NEED_CUDA=1
export TF_CUDA_CLANG=1
export CLANG_CUDA_COMPILER_PATH="/usr/lib/llvm-17/bin/clang"
export HERMETIC_CUDA_VERSION=12.6.0
export HERMETIC_CUDNN_VERSION=9.3.0 
export HERMETIC_CUDA_COMPUTE_CAPABILITIES=8.7

./oss_scripts/run_build.sh

pip3 install --verbose --no-cache-dir /opt/tensorflow-text/tensorflow_text-*.whl

twine upload --verbose /opt/tensorflow-text/tensorflow_text-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
