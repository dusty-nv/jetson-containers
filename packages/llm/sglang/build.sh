#!/usr/bin/env bash
set -ex

pip3 install compressed-tensors decord

REPO_URL="https://github.com/sgl-project/sglang"
REPO_DIR="/opt/sglang"

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" || echo "0.0")

# Convert to comparable float
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
CUDA_NUMERIC=$(( CUDA_MAJOR * 100 + CUDA_MINOR ))

# Clear all feature flags to defaults
export SGL_KERNEL_ENABLE_BF16=0
export SGL_KERNEL_ENABLE_FP8=0
export SGL_KERNEL_ENABLE_FP4=0
export SGL_KERNEL_ENABLE_SM90A=0
export SGL_KERNEL_ENABLE_SM100A=0
export SGL_KERNEL_ENABLE_SM103A=0
export SGL_KERNEL_ENABLE_SM110A=0
export SGL_KERNEL_ENABLE_FA3=1  # Always enabled

# Activate flags based on CUDA version
if (( CUDA_NUMERIC >= 1300 )); then
  echo "CUDA >= 13.0"
  export SGL_KERNEL_ENABLE_SM110A=1

elif (( CUDA_NUMERIC >= 1209 )); then
  echo "CUDA >= 12.9"
  export SGL_KERNEL_ENABLE_SM103A=1

elif (( CUDA_NUMERIC >= 1208 )); then
  echo "CUDA >= 12.8"
  export SGL_KERNEL_ENABLE_BF16=1
  export SGL_KERNEL_ENABLE_FP8=1
  export SGL_KERNEL_ENABLE_FP4=1
  export SGL_KERNEL_ENABLE_SM90A=1
  export SGL_KERNEL_ENABLE_SM100A=1

else
  echo "CUDA < 12.8"
  export SGL_KERNEL_ENABLE_BF16=1  # Only BF16 enabled
fi


echo "Building SGLang ${SGLANG_VERSION} for ${PLATFORM}"

git clone --recursive --depth=1 --branch=v${SGLANG_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR
echo "Building SGL-KERNEL"
cd $REPO_DIR/sgl-kernel/

# export MAX_JOBS="$(nproc)" this breaks with actual flash-attention
if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" ]]; then
    export MAX_JOBS=2
else
    export MAX_JOBS="$(nproc)"
fi

# 🔧 Build step for sgl‑kernel
echo "🔨  Building sgl‑kernel…"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS

echo "🚀  Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
pip3 wheel . --no-deps --wheel-dir $PIP_WHEEL_DIR
pip3 install $PIP_WHEEL_DIR/sgl*.whl
cd $REPO_DIR
if test -f "python/sglang/srt/utils.py"; then
    sed -i '/return min(memory_values)/s/.*/        return None/' python/sglang/srt/utils.py
    sed -i '/if not memory_values:/,+1d' python/sglang/srt/utils.py
fi

# 🔧 Build sglang
echo "🔨  Building sglang…"
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

if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" ]]; then
    export MAX_JOBS=6
else
    export MAX_JOBS="$(nproc)"
fi
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "🚀  Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

pip3 wheel '.[all]' --wheel-dir $PIP_WHEEL_DIR
pip3 install $PIP_WHEEL_DIR/sglang*.whl
cd /
echo "🔨  Installing gemlite…"
pip3 install gemlite
twine upload --verbose $PIP_WHEEL_DIR/sgl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
