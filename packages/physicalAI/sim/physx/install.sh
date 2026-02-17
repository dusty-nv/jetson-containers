#!/usr/bin/env bash
set -ex
echo "Building PhysX ${PHYSX_VERSION}"

git clone --recursive --depth=1 https://github.com/NVIDIA-Omniverse/PhysX /opt/PhysX
cd /opt/PhysX/physx
# Generate projects (first configure)
bash generate_projects.sh linux-aarch64-clang
# Reconfigure the release build dir to override the cached CUDA arch list
cd /opt/PhysX/physx/compiler/linux-aarch64-clang-release
cmake -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" .
make -j"$(nproc)"
make install
