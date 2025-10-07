#!/usr/bin/env bash
set -ex

echo "Building tilelang ${TILELANG_VERSION}"

apt-get update
apt-get install -y python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
git clone --recursive --depth=1 --branch=v${TILELANG_VERSION} https://github.com/tile-ai/tilelang /opt/tilelang ||
git clone --recursive --depth=1 https://github.com/tile-ai/tilelang  /opt/tilelang

cd /opt/tilelang


# export MAX_JOBS="$(nproc)" this breaks with actual tilelang
if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
    export MAX_JOBS=6
else
    export MAX_JOBS="$(nproc)"
fi
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
uv pip install -U -r requirements.txt
uv pip install -U -r requirements-build.txt

mkdir -p /opt/tilelang/build
cp /opt/tilelang/3rdparty/tvm/cmake/config.cmake build
cd /opt/tilelang/build
# echo "set(USE_LLVM ON)"  # set USE_LLVM to ON if using LLVM
{
  echo "set(USE_LLVM ON)";
  echo "set(USE_CUBLAS ON)";
  echo "set(USE_CUDNN ON)";
  echo "set(USE_CUDA ON)";
  echo "set(USE_CUTLASS ON)";
  echo "set(USE_THRUST ON)";
  echo "set(USE_NCCL ON)";
  echo "set(CMAKE_CUDA_ARCHITECTURES ${CUDAARCHS})";
} >> config.cmake
# or echo "set(USE_ROCM ON)" >> config.cmake to enable ROCm runtime
cmake ..
make -j "$MAX_JOBS"
MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
uv build --wheel . -v --no-deps --out-dir /opt/tilelang/wheels/

ls /opt
cd /

uv pip install /opt/tilelang/wheels/tilelang*.whl

twine upload --verbose /opt/tilelang/wheels/tilelang*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
