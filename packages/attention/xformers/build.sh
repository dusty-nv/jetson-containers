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
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

# https://github.com/Dao-AILab/flash-attention/blob/dc742f2c47baa4b15cc33e6a2444f33d02c0a6d4/setup.py#L59-L66
# We cannot compare the `1` to `TRUE` here as -----> [ "1" = "TRUE" ] && echo "Equal" || echo "Not equal"
MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
FLASH_ATTENTION_FORCE_BUILD="TRUE" \
FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" \
FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" \
XFORMERS_MORE_DETAILS=1 \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/xformers/wheels

pip3 install /opt/xformers/wheels/*.whl

twine upload --verbose /opt/xformers/wheels/xformers*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
