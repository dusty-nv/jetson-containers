#!/usr/bin/env bash
set -ex

pip3 install compressed-tensors decord

REPO_URL="https://github.com/sgl-project/sglang"
REPO_DIR="/opt/sglang"

ARCH="$(uname -m)"
ARCH_SED="s|x86_64|$ARCH|g" 
PLATFORM="$ARCH-linux"

export SGL_KERNEL_ENABLE_BF16=1
export SGL_KERNEL_ENABLE_FP8=1
export SGL_KERNEL_ENABLE_FP4=1
export SGL_KERNEL_ENABLE_SM90A=1
export SGL_KERNEL_ENABLE_SM100A=1
export USE_CUDNN=1

echo "Building SGLang ${SGLANG_VERSION} for ${PLATFORM}"

git clone --recursive --depth=1 --branch=v${SGLANG_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR
echo "Building SGL-KERNEL"
cd $REPO_DIR/sgl-kernel/

# export MAX_JOBS="$(nproc)" this breaks with actual flash-attention
export MAX_JOBS=3
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"


pip3 wheel . --no-deps --wheel-dir $PIP_WHEEL_DIR
pip3 install $PIP_WHEEL_DIR/sgl*.whl

cd $REPO_DIR

# Patch si hace falta
if test -f "python/sglang/srt/utils.py"; then
    sed -i '/return min(memory_values)/s/.*/        return None/' python/sglang/srt/utils.py
    sed -i '/if not memory_values:/,+1d' python/sglang/srt/utils.py
fi

# Build SGLang
cd $REPO_DIR/python

sed -i '/torchao/d' pyproject.toml
sed -i '/flashinfer_python/d' pyproject.toml
sed -i '/sgl-kernel/d' pyproject.toml
sed -i '/vllm/d' pyproject.toml
sed -i '/torch/d' pyproject.toml
sed -i '/torchvision/d' pyproject.toml
sed -i '/xgrammar/d' pyproject.toml
sed -i 's/==/>=/g' pyproject.toml


echo "Patched $REPO_DIR/python/pyproject.toml"
cat pyproject.toml

# export MAX_JOBS="$(nproc)" this breaks with actual flash-attention
export MAX_JOBS=6
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

pip3 wheel '.[all]' --wheel-dir $PIP_WHEEL_DIR
pip3 install $PIP_WHEEL_DIR/sglang*.whl

cd /

# Gemlite
pip3 install gemlite
twine upload --verbose $PIP_WHEEL_DIR/sgl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
