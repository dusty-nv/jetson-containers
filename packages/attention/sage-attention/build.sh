#!/usr/bin/env bash
set -ex

echo "Building SageAttention ${SAGE_ATTENTION_VERSION}"

git clone --depth=1 --branch=v${SAGE_ATTENTION_VERSION} https://github.com/johnnynunez/SageAttention /opt/sage-attention ||
git clone --depth=1 https://github.com/johnnynunez/SageAttention /opt/sage-attention

cd /opt/sage-attention

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/sage-attention/wheels

ls /opt/sage-attention/wheels
cd /

pip3 install /opt/sage-attention/wheels/sageattention*.whl

twine upload --verbose /opt/sage-attention/wheels/sage-attention*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
