#!/usr/bin/env bash
set -ex

echo "Building cccl ${CCCL_VERSION}"

REPO_URL="https://github.com/NVIDIA/cccl.git"
REPO_DIR="/opt/cuda_cccl"

git clone --recursive --depth=1 --branch=v${CCCL_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}


cd $REPO_DIR
cmake --preset install -DCMAKE_INSTALL_PREFIX=/usr/local/
cd build/install
ninja install
cd /opt/cuda_cccl/python/cuda_cccl
uv build --wheel --out-dir $REPO_DIR/wheels --verbose .
uv pip install /opt/cuda_cccl/wheels/cuda_cccl-*.whl

twine upload --verbose /opt/cuda_cccl/wheels/cuda_cccl-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
