#!/usr/bin/env bash
set -ex

echo "Building SpargeAttention ${SPARGE_ATTENTION_VERSION}"

git clone --depth=1 --branch=v${SPARGE_ATTENTION_VERSION} https://github.com/johnnynunez/SpargeAttn /opt/sparge-attention ||
git clone --depth=1 https://github.com/johnnynunez/SpargeAttn /opt/sparge-attention

cd /opt/sparge-attention

sed -i '1i#include <assert.h>' /usr/local/cuda/include/cuda_fp8.hpp

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/sparge-attention/wheels

ls /opt/sparge-attention/wheels
cd /

uv pip install /opt/sparge-attention/wheels/spas_sage_attn*.whl

twine upload --verbose /opt/sparge-attention/wheels/spas_sage_attn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
