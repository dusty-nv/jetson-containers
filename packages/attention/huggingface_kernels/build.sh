#!/usr/bin/env bash
set -ex

echo "Building Hugging-Face-Kernels ${KERNELS_VERSION}"

git clone --depth=1 --branch=v${KERNELS_VERSION} https://github.com/huggingface/kernels /opt/huggingface_kernels ||
git clone --depth=1 https://github.com/huggingface/kernels /opt/huggingface_kernels

cd /opt/huggingface_kernels

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
uv build --wheel . -v --out-dir /opt/huggingface_kernels/wheels/

ls /opt
cd /opt/

uv pip install /opt/huggingface_kernels/wheels/kernels*.whl
twine upload --verbose /opt/huggingface_kernels/wheels/kernels*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
