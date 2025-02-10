#!/usr/bin/env bash
set -ex
# Clone the repository if it doesn't exist
if [ ! -d /opt/glomap ]; then
    echo "Cloning glomap version ${GLOMAP_VERSION}"
    git clone --branch=v${GLOMAP_VERSION} --depth=1 --recursive https://github.com/colmap/glomap /opt/glomap || \
    git clone --depth=1 --recursive https://github.com/colmap/glomap /opt/glomap
fi

cd /opt/glomap
mkdir build
cd build
mkdir -p /build
cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS}" -DCMAKE_INSTALL_PREFIX=/build/glomap
ninja install -j $(nproc)
ldconfig