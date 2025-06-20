#!/usr/bin/env bash
set -ex

echo "Building apex ${APEX_VERSION}"

git clone --depth=1 --branch=v${APEX_VERSION} https://github.com/NVIDIA/apex /opt/apex ||
git clone --depth=1 https://github.com/NVIDIA/apex /opt/apex

cd /opt/apex
export MAX_JOBS=$(nproc)
NVCC_APPEND_FLAGS="--threads 12" pip3 wheel . -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext" . -w /opt/apex/wheels/
pip3 install /opt/apex/wheels/apex*.whl

twine upload --verbose /opt/apex/wheels/apex*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
