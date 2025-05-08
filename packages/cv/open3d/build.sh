#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
set -x

# Ensure required variables are set
: "${OPEN3D_VERSION:?OPEN3D_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

REPO_URL="https://github.com/johnnynunez/Open3D"
REPO_DIR="/opt/open3d"

echo "Building Open3D ${OPEN3D_VERSION}"

# Clone either the tagged release or fallback to default branch
if git clone --recursive --depth 1 --branch "v${OPEN3D_VERSION}" \
    "${REPO_URL}" "${REPO_DIR}"
then
  echo "Cloned v${OPEN3D_VERSION}"
else
  echo "Tagged branch not found; cloning default branch"
  git clone --recursive --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}" || exit 1

mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_CUDA_MODULE=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DGLIBCXX_USE_CXX11_ABI=OFF \
      -DBUILD_PYTORCH_OPS=OFF \
      -DBUILD_TENSORFLOW_OPS=OFF \
      ..

# -DBUNDLE_OPEN3D_ML=ON \
# -DOPEN3D_ML_ROOT=https://github.com/isl-org/Open3D-ML.git \
export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS

make -j$(nproc)
make install-pip-package -j$(nproc)
make install -j$(nproc)
cd "${REPO_DIR}" || exit 1
cp build/lib/python_package/pip_package/*.whl /opt/open3d/
# Try uploading; ignore failure
twine upload --verbose "/opt/open3d/open3d"*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
