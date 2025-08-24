#!/usr/bin/env bash
set -ex

echo "Building tilelang ${TILELANG_VERSION}"

apt-get update
apt-get install -y python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
git clone --depth=1 --branch=v${TILELANG_VERSION} https://github.com/tile-ai/tilelang /opt/tilelang ||
git clone --depth=1 https://github.com/tile-ai/tilelang  /opt/tilelang

cd /opt/tilelang


# export MAX_JOBS="$(nproc)" this breaks with actual tilelang
if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
    export MAX_JOBS=6
else
    export MAX_JOBS="$(nproc)"
fi
mkdir build
cd build
cmake .. -DTVM_PREBUILD_PATH=/opt/tvm/  # e.g., /workspace/tvm/build
make -j 16

export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
pip3 install -U -r requirements.txt
pip3 install -U -r requirements-build.txt

MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
pip3 wheel . -v --no-deps -w /opt/tilelang/wheels/

ls /opt
cd /

pip3 install /opt/tilelang/wheels/tilelang*.whl

twine upload --verbose /opt/tilelang/wheels/tilelang*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
