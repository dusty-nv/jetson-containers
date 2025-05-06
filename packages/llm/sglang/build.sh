#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
set -x

# Ensure required variables are set
: "${SGLANG_VERSION:?SGLANG_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

# Install Python deps
pip3 install compressed-tensors decord

REPO_URL="https://github.com/sgl-project/sglang"
REPO_DIR="/opt/sglang"

echo "Building SGLang ${SGLANG_VERSION}"

# Clone either the tagged release or fallback to default branch
if git clone --recursive --depth 1 --branch "v${SGLANG_VERSION}" \
    "${REPO_URL}" "${REPO_DIR}"
then
  echo "Cloned v${SGLANG_VERSION}"
else
  echo "Tagged branch not found; cloning default branch"
  git clone --recursive --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

echo "Building SGL-KERNEL"
cd "${REPO_DIR}/sgl-kernel" || exit 1

# Check CUDA version (nvcc must be on PATH)
CUDA_VERSION=$(nvcc --version \
  | grep -oP 'release \K[0-9]+\.[0-9]+' \
  || echo '0.0')

CUDA_MAJOR=${CUDA_VERSION%%.*}
CUDA_MINOR=${CUDA_VERSION#*.}
CUDA_NUMERIC=$(( CUDA_MAJOR * 100 + CUDA_MINOR ))

# Clear all feature flags to defaults
export SGL_KERNEL_ENABLE_BF16=0
export SGL_KERNEL_ENABLE_FP8=0
export SGL_KERNEL_ENABLE_FP4=0
export SGL_KERNEL_ENABLE_SM90A=0
export SGL_KERNEL_ENABLE_SM100A=0
export SGL_KERNEL_ENABLE_SM103A=0
export SGL_KERNEL_ENABLE_SM110A=0
export SGL_KERNEL_ENABLE_FA3=0  # Always enabled via CMake

if (( CUDA_NUMERIC >= 1300 )); then
  echo "CUDA >= 13.0"
  export SGL_KERNEL_ENABLE_SM110A=1

elif (( CUDA_NUMERIC >= 1209 )); then
  echo "CUDA >= 12.9"
  export SGL_KERNEL_ENABLE_SM103A=1

elif (( CUDA_NUMERIC >= 1208 )); then
  echo "CUDA >= 12.8 (SBSA=${IS_SBSA:-})"
  export SGL_KERNEL_ENABLE_BF16=1
  export SGL_KERNEL_ENABLE_FP8=1
  export SGL_KERNEL_ENABLE_FP4=1
  export SGL_KERNEL_ENABLE_SM100A=1

  sed -i \
    -e '/-gencode=arch=compute_75,code=sm_75/d' \
    -e '/"-O3"/a    "-gencode=arch=compute_87,code=sm_87"' \
    CMakeLists.txt

  if [[ "${IS_SBSA:-}" == "1" || "${IS_SBSA,,}" == "true" ]]; then
    export SGL_KERNEL_ENABLE_SM90A=1
    export SGL_KERNEL_ENABLE_FA3=1
  fi

else
  echo "CUDA < 12.8"
  export SGL_KERNEL_ENABLE_BF16=1  # Only BF16 enabled

  sed -i \
    -e '/-gencode=arch=compute_75,code=sm_75/d' \
    -e '/-gencode=arch=compute_80,code=sm_80/d' \
    -e '/-gencode=arch=compute_89,code=sm_89/d' \
    -e '/-gencode=arch=compute_90,code=sm_90/d' \
    -e '/"-O3"/a    "-gencode=arch=compute_87,code=sm_87"' \
    CMakeLists.txt
fi

if [[ "${IS_SBSA:-}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
  # On non-SBSA builds, strip out any automatic FA3=ON
  sed -i '/^[[:space:]]*set(SGL_KERNEL_ENABLE_FA3[[:space:]]\+ON)/d' CMakeLists.txt
fi

echo "Patched sgl-kernel/CMakeLists.txt"
cat CMakeLists.txt

# 🔧 Build step for sgl-kernel
echo "🔨  Building sgl-kernel…"
if [[ "${IS_SBSA:-}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
  export CORES=2
else
  export CORES=32  # GH200
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CORES}"

echo "🚀  Building with MAX_JOBS=${CORES} and CMAKE_BUILD_PARALLEL_LEVEL=${CORES}"
MAX_JOBS="${CORES}" \
CMAKE_BUILD_PARALLEL_LEVEL="${CORES}" \
pip3 wheel . --no-deps --wheel-dir "${PIP_WHEEL_DIR}"
pip3 install "${PIP_WHEEL_DIR}/sgl"*.whl

cd "${REPO_DIR}" || exit 1

# Patch utils.py if present
UTILS_PATH="python/sglang/srt/utils.py"
if [[ -f "${UTILS_PATH}" ]]; then
  sed -i \
    -e '/return min(memory_values)/s/.*/        return None/' \
    -e '/if not memory_values:/,+1d' \
    "${UTILS_PATH}"
fi

# 🔧 Build sglang (Python package)
echo "🔨  Building sglang…"
cd "${REPO_DIR}/python" || exit 1

# Remove unwanted dependencies in pyproject.toml
for pkg in torchao flashinfer_python sgl-kernel vllm torch torchvision xgrammar; do
  sed -i "/${pkg}/d" pyproject.toml
done
# Relax any strict version pins
sed -i 's/==/>=/g' pyproject.toml

echo "Patched ${REPO_DIR}/python/pyproject.toml"
cat pyproject.toml

if [[ -z "${IS_SBSA:-}" || "${IS_SBSA}" == "0" ]]; then
  export CORES=6
else
  export CORES=32  # GH200
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CORES}"

echo "🚀  Building with MAX_JOBS=${CORES} and CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"
MAX_JOBS="${CORES}" \
CMAKE_BUILD_PARALLEL_LEVEL="${CORES}" \
pip3 wheel '.[all]' --wheel-dir "${PIP_WHEEL_DIR}"
pip3 install "${PIP_WHEEL_DIR}/sgl"*.whl

cd / || exit 1

echo "🔨  Installing gemlite…"
pip3 install gemlite

# Try uploading; ignore failure
twine upload --verbose "${PIP_WHEEL_DIR}/sgl"*.whl \
  || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
