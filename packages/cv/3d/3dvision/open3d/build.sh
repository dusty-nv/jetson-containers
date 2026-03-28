#!/usr/bin/env bash
set -ex

echo "Building Open3D ${OPEN3D_VERSION} with CUDA and ML_OPS"

OPEN3D_SRC="${OPEN3D_SRC:-/opt/open3d}"
OPEN3D_BUILD_DIR="${OPEN3D_SRC}/build"
PIP_WHEEL_DIR="${PIP_WHEEL_DIR:-/opt/wheels}"

TORCH_CXX11_ABI=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" 2>/dev/null || echo "1")
echo "PyTorch CXX11 ABI: ${TORCH_CXX11_ABI}"

git clone --branch=v${OPEN3D_VERSION} --depth=1 --recursive \
    https://github.com/isl-org/Open3D "${OPEN3D_SRC}" || \
git clone --depth=1 --recursive \
    https://github.com/isl-org/Open3D "${OPEN3D_SRC}"

cd "${OPEN3D_SRC}"

if [ -f util/install_deps_ubuntu.sh ]; then
    yes | util/install_deps_ubuntu.sh || true
fi

mkdir -p "${OPEN3D_BUILD_DIR}"
cd "${OPEN3D_BUILD_DIR}"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDAARCHS}" \
    -DBUILD_CUDA_MODULE=ON \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_OPEN3D_ML=ON \
    -DOPEN3D_ML_ROOT=https://github.com/isl-org/Open3D-ML.git \
    -DGLIBCXX_USE_CXX11_ABI="${TORCH_CXX11_ABI}" \
    -DBUILD_PYTHON_MODULE=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    ..

make -j$(nproc)
make install
ldconfig

make -j$(nproc) pip-package

mkdir -p "${PIP_WHEEL_DIR}"
cp lib/python_package/pip_package/open3d*.whl "${PIP_WHEEL_DIR}"/

uv pip install "${PIP_WHEEL_DIR}"/open3d*.whl
uv pip show open3d

twine upload --verbose "${PIP_WHEEL_DIR}"/open3d*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

python3 -c "import open3d; print('Open3D version:', open3d.__version__)"
