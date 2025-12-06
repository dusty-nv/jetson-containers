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
export CLANG_CUDA_COMPILER_PATH="/usr/lib/llvm-21/bin/clang"
export HERMETIC_CUDA_VERSION=13.1.0
export HERMETIC_CUDNN_VERSION=9.17.0
export HERMETIC_CUDA_COMPUTE_CAPABILITIES=8.7,8.9,9.0,11.0,12.0,12.1
export TF_VERSION=${TENSORFLOW_TEXT_VERSION}
./oss_scripts/configure.sh
./oss_scripts/prepare_tf_dep.sh

bazel run --enable_runfiles //oss_scripts/pip_package:build_pip_package -- "$(realpath .)" --action_env CLANG_CUDA_COMPILER_PATH="/usr/lib/llvm-21/bin/clang" --config=cuda_clang --repo_env=WHEEL_NAME=tensorflow --config=cuda --config=cuda_wheel --config=nonccl --copt=-Wno-sign-compare --copt=-Wno-gnu-offsetof-extensions --copt=-Wno-error=unused-command-line-argument

uv pip install /opt/tensorflow-text/tensorflow_text-*.whl

twine upload --verbose /opt/tensorflow-text/tensorflow_text-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# clean all files
bazel clean --expunge
