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

# Fix the package version
codebase_version=$(grep -E '^__version__\s*=\s*"' flash_attn/__init__.py | sed -E 's/^__version__\s*=\s*"([^"]+)".*/\1/')
echo "Codebase version: ${codebase_version}"
echo "Wanted version: ${FLASH_ATTENTION_VERSION}"

if [ "$codebase_version" != "${FLASH_ATTENTION_VERSION}" ]; then
    sed -i \
        -e "s|^__version__ = \".*\"|__version__ = \"${FLASH_ATTENTION_VERSION}\"|" \
        flash_attn/__init__.py
    cat flash_attn/__init__.py
fi

# export MAX_JOBS="$(nproc)" this breaks with actual flash-attention
if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
    export MAX_JOBS=6
else
    export MAX_JOBS="$(nproc)"
fi
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

# https://github.com/Dao-AILab/flash-attention/blob/dc742f2c47baa4b15cc33e6a2444f33d02c0a6d4/setup.py#L59-L66
# We cannot compare the `1` to `TRUE` here as -----> [ "1" = "TRUE" ] && echo "Equal" || echo "Not equal"
MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
FLASH_ATTENTION_FORCE_BUILD="TRUE" \
FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" \
FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" \
pip3 wheel . -v --no-deps -w /opt/flash-attention/wheels/

ls /opt
cd /

pip3 install /opt/flash-attention/wheels/flash_attn*.whl
#pip3 show flash-attn && python3 -c 'import flash_attn'

twine upload --verbose /opt/flash-attention/wheels/flash_attn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
