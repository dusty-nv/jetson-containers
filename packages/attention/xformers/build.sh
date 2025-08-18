#!/usr/bin/env bash
set -ex

echo "Building xformers ${XFORMERS_VERSION}"

git clone --branch=v${XFORMERS_VERSION} --depth=1 --recursive https://github.com/facebookresearch/xformers /opt/xformers ||
git clone --depth=1 --recursive https://github.com/facebookresearch/xformers /opt/xformers

cd /opt/xformers

if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
    export MAX_JOBS=6
else
    export MAX_JOBS="$(nproc)"
fi
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
export NVCC_THREADS=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS"

MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
XFORMERS_DISABLE_FLASH_ATTN=1 \
XFORMERS_MORE_DETAILS=1 \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/xformers/wheels

pip3 install /opt/xformers/wheels/*.whl

twine upload --verbose /opt/xformers/wheels/xformers*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
