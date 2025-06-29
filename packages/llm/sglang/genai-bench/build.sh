#!/usr/bin/env bash
set -x

# Ensure required variables are set
: "${GENAI_BENCH_VERSION:?GENAI_BENCH_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

# Install Python deps
pip3 install compressed-tensors decord2

REPO_URL="https://github.com/sgl-project/genai-bench"
REPO_DIR="/opt/genai-bench"

echo "Building genai-bench ${GENAI_BENCH_VERSION}"

# Clone either the tagged release or fallback to default branch
echo "Tagged branch not found; cloning default branch"
git clone --recursive --depth 1 "${REPO_URL}" "${REPO_DIR}"

pip3 install --no-cache-dir ninja setuptools wheel numpy uv scikit-build-core
echo "Building GENAI_BENCH"
cd "${REPO_DIR}" || exit 1

make install
pip3 wheel --wheel-dir=/opt --no-deps --verbose .
pip3 install "${PIP_WHEEL_DIR}/genai-bench"*.whl
cd "${REPO_DIR}" || exit 1

twine upload --verbose "${PIP_WHEEL_DIR}/genai-bench"*.whl \
  || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
