#!/usr/bin/env bash
set -ex

echo "Building FlashAttention ${FLASH_ATTENTION_VERSION}"

git clone --depth=1 --branch=v${FLASH_ATTENTION_VERSION} https://github.com/Dao-AILab/flash-attention /opt/flash-attention ||
git clone --depth=1 https://github.com/Dao-AILab/flash-attention /opt/flash-attention

cd /opt/flash-attention

# Generate the diff dynamically
python3 /tmp/flash-attention/generate_diff.py
git apply /tmp/flash-attention/patch.diff
git diff
git status

# export MAX_JOBS="$(nproc)" this breaks with actual flash-attention
if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
    export MAX_JOBS=6
else
    export MAX_JOBS="$(nproc)"
fi
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
FLASH_ATTENTION_FORCE_BUILD=1 \
FLASH_ATTENTION_FORCE_CXX11_ABI=0 \
FLASH_ATTENTION_SKIP_CUDA_BUILD=0 \
pip3 wheel . -v --no-deps -w /opt/flash-attention/wheels/

ls /opt
cd /

pip3 install /opt/flash-attention/wheels/flash_attn*.whl
#pip3 show flash-attn && python3 -c 'import flash_attn'

twine upload --verbose /opt/flash-attention/wheels/flash_attn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
