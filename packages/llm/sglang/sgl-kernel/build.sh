#!/usr/bin/env bash
set -x

# Ensure required variables are set
: "${SGL_KERNEL_VERSION:?SGL_KERNEL_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

# Install Python deps
pip3 install compressed-tensors decord2 ninja setuptools wheel numpy uv scikit-build-core

REPO_URL="https://github.com/sgl-project/sglang"
REPO_DIR="/opt/sglang"

echo "Building SGLang ${SGL_KERNEL_VERSION}"

# Clone either the tagged release or fallback to default branch
echo "Tagged branch not found; cloning default branch"
git clone --recursive --depth 1 "${REPO_URL}" "${REPO_DIR}"
sed -i 's/==/>=/g' pyproject.toml

# 🔧 Build step for sgl-kernel
echo "🔨  Building sgl-kernel…"
if [[ "${IS_SBSA:-}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
  export CORES=2
else
  export CORES=32  # GH200
  # Set the correct include path to point to the CCCL directory
  export CPLUS_INCLUDE_PATH=$(echo /usr/local/cuda-*/targets/sbsa-linux/include/cccl | head -n1)
fi

echo "🚀  Building with MAX_JOBS=${CORES} and CMAKE_BUILD_PARALLEL_LEVEL=${CORES}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
MAX_JOBS="${CORES}" \
CMAKE_BUILD_PARALLEL_LEVEL="${CORES}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
pip3 wheel . --no-build-isolation --wheel-dir "${PIP_WHEEL_DIR}"
pip3 install "${PIP_WHEEL_DIR}/sgl"*.whl
cd "${REPO_DIR}" || exit 1

twine upload --verbose "${PIP_WHEEL_DIR}/sgl"*.whl \
  || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
