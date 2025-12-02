#!/usr/bin/env bash
set -ex

echo "Building JVP-FlashAttention ${JVP_FLASH_ATTENTION_VERSION}"

git clone --depth=1 --branch=v${JVP_FLASH_ATTENTION_VERSION} https://github.com/amorehead/jvp_flash_attention /opt/jvp-flash-attention ||
git clone --depth=1 https://github.com/amorehead/jvp_flash_attention /opt/jvp-flash-attention

cd /opt/jvp-flash-attention

export NVCC_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
uv build --wheel . -v --no-build-isolation --out-dir /opt/jvp-flash-attention/wheels/

ls /opt
cd /

uv pip install /opt/jvp-flash-attention/wheels/jvp_flash_attention*.whl

twine upload --verbose /opt/jvp-flash-attention/wheels/jvp_flash_attention*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
