#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
if [ ! -d /opt/pycolmap ]; then
    echo "Cloning pycolmap version ${PYCOLMAP_VERSION}"
    git clone --branch=v${PYCOLMAP_VERSION} --depth=1 --recursive https://github.com/colmap/colmap /opt/colmap || \
    git clone --depth=1 --recursive https://github.com/colmap/colmap /opt/colmap
fi

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/colmap && \
mkdir build && \
cd build && \
cmake .. -DCUDA_ENABLED=ON \
            -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} && \
make -j $(nproc) && \
make install && \
cd /opt/colmap/pycolmap && \
pip3 wheel . -w /opt/pycolmap/wheels --verbose
pip3 install /opt/pycolmap/wheels/pycolmap-*.whl
ldconfig

# Verify the contents of the /opt directory
ls /opt/pycolmap/wheels

# Return to the root directory
cd /

pip3 install --no-cache-dir --verbose /opt/pycolmap/wheels/pycolmap*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/pycolmap/wheels/pycolmap*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
