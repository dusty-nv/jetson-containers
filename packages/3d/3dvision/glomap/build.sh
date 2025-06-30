#!/usr/bin/env bash
set -ex

GLOMAP_SRC="${GLOMAP_SRC:-/opt/glomap}"

if [ ! -d $GLOMAP_SRC ]; then
    echo "Cloning glomap version ${GLOMAP_VERSION}"
    git clone --branch=v${GLOMAP_VERSION} --depth=1 --recursive https://github.com/colmap/glomap $GLOMAP_SRC || \
    git clone --depth=1 --recursive https://github.com/colmap/glomap $GLOMAP_SRC
fi

cd $GLOMAP_SRC

# https://github.com/colmap/glomap/issues/182
sed -i 's|-Werror||g' glomap/CMakeLists.txt

# configure source tree with cmake
mkdir -p build/dist
cd build
cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS}" -DCMAKE_INSTALL_PREFIX=$GLOMAP_SRC/build/dist

# https://github.com/colmap/glomap/issues/182#issuecomment-2816872337
sed -i 's|-Werror||g' _deps/poselib-src/CMakeLists.txt

# build GLOMAP module
ninja install -j $(nproc)
ldconfig