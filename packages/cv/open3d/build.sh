#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
set -x

# Ensure required variables are set
: "${OPEN3D_VERSION:?OPEN3D_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

REPO_URL="https://github.com/isl-org/Open3D"
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

bash util/install_deps_ubuntu.sh
mkdir build
cd build
cmake -DBUILD_CUDA_MODULE=ON \
      -DGLIBCXX_USE_CXX11_ABI=OFF \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_OPEN3D_ML=ON \
      -DOPEN3D_ML_ROOT=https://github.com/isl-org/Open3D-ML.git \
      ..
make install # install open3d C++
make -j$(nproc) pip-package

# Try uploading; ignore failure
twine upload --verbose "opt/open3d/open3d"*.whl \
  || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
