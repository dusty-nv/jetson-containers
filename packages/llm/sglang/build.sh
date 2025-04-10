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
export SGL_KERNEL_ENABLE_FP4=0
export SGL_KERNEL_ENABLE_SM90A=1
export SGL_KERNEL_ENABLE_SM100A=0
export USE_CUDNN=1
# export MAX_JOBS="$(nproc)" this breaks with actual flash-attention
export MAX_JOBS=6

echo "Building SGLang ${SGLANG_VERSION} for ${PLATFORM}"

git clone --recursive --depth=1 --branch=v${SGLANG_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR

cd $REPO_DIR

# Clean deps
sed -i '/sgl-kernel/d' python/pyproject.toml
sed -i '/flashinfer/d' python/pyproject.toml
sed -i '/xgrammar/d' python/pyproject.toml
sed -i '/"torch==2\.5\.1",/d' python/pyproject.toml

# Patch arch in init
sed -i $ARCH_SED sgl-kernel/python/sgl_kernel/__init__.py

echo "Building SGL-KERNEL"
cd $REPO_DIR/sgl-kernel/

sed -i '/"torch==2\.5\.1",/d' pyproject.toml
sed -i 's|"torch==.*"|"torch"|g' pyproject.toml

pip3 install "cmake<4"
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=87;101" pip3 wheel . --no-deps --wheel-dir $PIP_WHEEL_DIR
pip3 install $PIP_WHEEL_DIR/sgl*.whl

cd $REPO_DIR

# Patch si hace falta
if test -f "python/sglang/srt/utils.py"; then
    sed -i '/return min(memory_values)/s/.*/        return None/' python/sglang/srt/utils.py
    sed -i '/if not memory_values:/,+1d' python/sglang/srt/utils.py
fi

# Build SGLang
cd $REPO_DIR/python

sed -i 's|"torchao.*"|"torchao"|g' pyproject.toml
sed -i 's|"sgl-kernel.*"|"sgl-kernel"|g' pyproject.toml
sed -i 's|"vllm.*"|"vllm"|g' pyproject.toml
sed -i 's|"torch=.*"|"torch"|g' pyproject.toml

echo "Patched $REPO_DIR/python/pyproject.toml"
cat pyproject.toml

pip3 install "cmake<4"
pip3 wheel '.[all]' --wheel-dir $PIP_WHEEL_DIR
pip3 install $PIP_WHEEL_DIR/sglang*.whl

cd /

# Gemlite
pip3 install gemlite

pip3 show sglang
python3 -c 'import sglang'

twine upload --verbose $PIP_WHEEL_DIR/sgl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"