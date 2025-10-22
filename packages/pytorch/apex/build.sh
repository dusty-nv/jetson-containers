#!/usr/bin/env bash
set -ex

echo "Building apex ${APEX_VERSION}"

git clone --depth=1 --branch=v${APEX_VERSION} https://github.com/NVIDIA/apex /opt/apex ||
git clone --depth=1 https://github.com/NVIDIA/apex /opt/apex

cd /opt/apex
export MAX_JOBS=$(nproc)
NVCC_APPEND_FLAGS="--threads 12" \
uv build --wheel \
  --no-cache \
  --no-build-isolation \
  -C=--build-option=--cpp_ext \
  -C=--build-option=--cuda_ext \
  --out-dir /opt/apex/wheels \
  .
uv pip install /opt/apex/wheels/apex*.whl

twine upload --verbose /opt/apex/wheels/apex*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
