#!/usr/bin/env bash
set -ex

echo "Building GPTQMODEL ${GPTQMODEL_VERSION}"

git clone --branch=v${GPTQMODEL_BRANCH} --depth=1 https://github.com/ModelCloud/GPTQModel /opt/gptmodel || \
git clone --depth=1 https://github.com/ModelCloud/GPTQModel /opt/gptmodel

cd /opt/gptmodel
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//;/ }"
export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
python3 setup.py --verbose bdist_wheel --dist-dir /opt/gptqmodel/wheels/

uv pip install /opt/gptqmodel/wheels/gptqmodel*.whl
uv pip show auto-gptq && python3 -c 'import gptqmodel'

twine upload --verbose /opt/gptqmodel/wheels/gptqmodel*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
