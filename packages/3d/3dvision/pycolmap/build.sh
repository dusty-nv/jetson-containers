#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
PYCOLMAP_SRC="${PYCOLMAP_SRC:-/opt/pycolmap}"

if [ ! -d $PYCOLMAP_SRC ]; then
    echo "Cloning pycolmap version ${PYCOLMAP_VERSION}"
    git clone --branch=v${PYCOLMAP_VERSION} --depth=1 --recursive https://github.com/colmap/colmap $PYCOLMAP_SRC || \
    git clone --depth=1 --recursive https://github.com/colmap/colmap $PYCOLMAP_SRC
fi

mkdir -p $PYCOLMAP_SRC/build
cd $PYCOLMAP_SRC/build

# Build base libraries
cmake \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} \
    ..
    
make -j $(nproc)
make install

# Build python wheel from source
cd $PYCOLMAP_SRC
export MAX_JOBS=$(nproc)

pip3 wheel . -w $PIP_WHEEL_DIR --verbose
pip3 install $PIP_WHEEL_DIR/pycolmap-*.whl

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/pycolmap*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
