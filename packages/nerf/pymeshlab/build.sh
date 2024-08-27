#!/usr/bin/env bash
set -ex

echo "Building pymeshlab ${PYMESHLAB_VERSION}"

git clone --branch=v${PYMESHLAB_VERSION} --depth=1 --recursive https://github.com/cnr-isti-vclab/PyMeshLab /opt/pymeshlab

cd /opt/pymeshlab

# Create a build directory
mkdir build
cd build

# Configure the project with CMake and Ninja
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-error" ..

# Build with Ninja
ninja

# Install the built package
ninja install

MAX_JOBS=$(nproc) \
pip3 wheel . -w /opt

ls /opt
cd /

pip3 install --no-cache-dir --verbose /opt/pymeshlab*.whl

twine upload --verbose /opt/pymeshlab*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"