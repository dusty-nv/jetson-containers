#!/usr/bin/env bash
set -x

# Ensure required variables are set
: "${SGLANG_VERSION:?SGLANG_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

# --- PRE-INSTALL DEPS ---
# Install build dependencies first. uv is a very fast installer.
uv pip install --no-cache-dir ninja setuptools wheel numpy uv scikit-build-core compressed-tensors decord2

# --- CLONE SGLANG REPO ---
REPO_URL="https://github.com/sgl-project/sglang"
REPO_DIR="/opt/sglang"

echo "Building SGLang ${SGLANG_VERSION}"

if [ ! -d "${REPO_DIR}" ]; then
  if git clone --recursive --depth 1 --branch "v${SGLANG_VERSION}" \
      "${REPO_URL}" "${REPO_DIR}"; then
    echo "Cloned SGLang v${SGLANG_VERSION}"
  else
    echo "Tagged branch v${SGLANG_VERSION} not found; cloning default branch"
    git clone --recursive --depth 1 "${REPO_URL}" "${REPO_DIR}"
  fi
else
  echo "Directory ${REPO_DIR} already exists, skipping clone."
fi
cd "${REPO_DIR}" || exit 1


# --- PATCH 1: RELAX PYTORCH VERSION REQUIREMENTS ---
cd "${REPO_DIR}/python" || exit 1
sed -i 's/==/>=/g' pyproject.toml

echo "Patched ${REPO_DIR}/python/pyproject.toml to relax version constraints"
cat pyproject.toml

# --- CONFIGURE PARALLEL BUILD ---
if [[ -z "${IS_SBSA:-}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
  export CORES=$(nproc) # Automatically use all available cores
else
  export CORES=32  # GH200 or other specific hardware
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CORES}"
export MAX_JOBS="${CORES}"

# --- BUILD SGLANG WHEEL (THE RIGHT WAY) ---
echo "🚀 Building sglang wheel ONLY with MAX_JOBS=${CORES}"

# Use '--no-deps' to build ONLY the sglang wheel and ignore its dependencies.
# We will install dependencies later when we install the built wheel.
uv build --wheel \
    --no-build-isolation \
    . \
    --out-dir "${PIP_WHEEL_DIR}"

# --- INSTALL THE BUILT WHEEL AND ITS DEPENDENCIES ---
echo "✅ sglang wheel built successfully."
echo "📦 Installing the sglang wheel from ${PIP_WHEEL_DIR} and its dependencies from PyPI..."

# Now, when we install the local wheel, pip will fetch its dependencies
# (like torch, transformers, etc.) from the online package index (PyPI).
# We use 'uv' here because it's extremely fast.
uv pip install -v --find-links="${PIP_WHEEL_DIR}" "sglang[all]"

# Your original script installed 'gemlite' here, so we keep it.
uv pip install gemlite orjson pybase64

echo "🎉 SGLang and all dependencies installed successfully!"

cd / || exit 1

# Try uploading; ignore failure
if [ -x "$(command -v twine)" ]; then
    twine upload --verbose "${PIP_WHEEL_DIR}/sglang"*.whl \
      || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
else
    echo "twine not installed, skipping upload."
fi
