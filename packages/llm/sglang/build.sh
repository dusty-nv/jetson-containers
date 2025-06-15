#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
set -x

# Ensure required variables are set
: "${SGLANG_VERSION:?SGLANG_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

# Install Python deps
pip3 install compressed-tensors decord2

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

pip3 install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core
echo "Building SGL-KERNEL"
cd "${REPO_DIR}/sgl-kernel" || exit 1
sed -i -E 's/(set[[:space:]]*\(ENABLE_BELOW_SM90)[[:space:]]+OFF/\1 ON/' CMakeLists.txt
sed -i -E 's/(message[[:space:]]*\([[:space:]]*STATUS[[:space:]]*")[^"]*(")/\1ACTIVATED\2/' CMakeLists.txt
# sed -i '/^        "-gencode=arch=compute_80,code=sm_80"/a\        "-gencode=arch=compute_87,code=sm_87"' CMakeLists.txt
# sed -i '/^            "-gencode=arch=compute_80,code=sm_80"/a\            "-gencode=arch=compute_87,code=sm_87"' CMakeLists.txt
sed -i 's/==/>=/g' pyproject.toml


# 🔧 Build step for sgl-kernel
echo "🔨  Building sgl-kernel…"
if [[ "${IS_SBSA:-}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
  export CORES=2
else
  export CORES=32  # GH200
fi

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

sed -i 's/==/>=/g' pyproject.toml

echo "Patched ${REPO_DIR}/python/pyproject.toml"
cat pyproject.toml

if [[ -z "${IS_SBSA:-}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
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
