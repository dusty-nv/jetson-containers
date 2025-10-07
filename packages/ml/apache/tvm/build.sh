#!/usr/bin/env bash
set -ex

echo "Building Apache TVM from source (commit=${TVM_COMMIT})"

export TVM_SRC_DIR=/opt/tvm

git clone --recursive https://github.com/apache/tvm.git ${TVM_SRC_DIR}
cd ${TVM_SRC_DIR}

if [ -n "${TVM_COMMIT}" ]; then
    git fetch origin ${TVM_COMMIT} || true
    git checkout ${TVM_COMMIT}
fi

git submodule update --init --recursive

mkdir -p build
cp cmake/config.cmake build/config.cmake

# Configure per instructions
{
  echo "set(USE_LLVM ON)";
  echo "set(USE_CUBLAS ON)";
  echo "set(USE_CUDNN ON)";
  echo "set(USE_CUDA ON)";
  echo "set(USE_CUTLASS ON)";
  echo "set(USE_THRUST ON)";
  echo "set(USE_NCCL ON)";
  echo "set(CMAKE_CUDA_ARCHITECTURES ${CUDAARCHS})";
} >> build/config.cmake

cd build
cmake ..
make -j$(nproc)


cd /opt/tvm/python
uv build --wheel --out-dir /opt/tvm/wheels .

uv pip install /opt/tvm/wheels/tvm-*.whl

twine upload --verbose dist/tvm-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
