#!/usr/bin/env bash
set -ex

echo "Building SageAttention ${SAGE_ATTENTION_VERSION}"

git clone --recursive --depth=1 --branch=v${SAGE_ATTENTION_VERSION} https://github.com/thu-ml/SageAttention /opt/sage-attention ||
git clone --recursive --depth=1 https://github.com/thu-ml/SageAttention /opt/sage-attention

cd /opt/sage-attention

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

export TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
uv build --wheel . -v --no-build-isolation --out-dir /opt/sage-attention/wheels/

ls /opt/sage-attention/wheels

# Do if Blackwell is enabled
cd /opt/sage-attention/sageattention3_blackwell/
uv build --wheel . -v --no-build-isolation --out-dir /opt/sage-attention/wheels/ || echo "Blackwell build failed, skipping..."

cd /

uv pip install /opt/sage-attention/wheels/sageattention*.whl
uv pip install /opt/sage-attention/wheels/sageattn3*.whl || echo "Blackwell wheel install failed, skipping..."

twine upload --verbose /opt/sage-attention/wheels/sage-attention*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/sage-attention/wheels/sageattn3*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
