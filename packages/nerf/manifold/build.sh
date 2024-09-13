#!/usr/bin/env bash
set -ex


echo "Cloning manifold version ${MANIFOLD_VERSION}"
git clone --branch=v${MANIFOLD_VERSION} --depth=1 --recursive https://github.com/elalish/manifold.git /opt/manifold

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/manifold && \
mkdir build && \
cd build && \
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DMANIFOLD_PYBIND=ON -DMANIFOLD_CROSS_SECTION=ON .. && \
make -j $(nproc) && \
cd /opt/manifold && pip3 wheel . -w /opt/manifold/wheels

# Verify the contents of the /opt directory
ls /opt/manifold/wheels

# Return to the root directory
cd /

pip3 install --no-cache-dir --verbose /opt/manifold/wheels/manifold3d*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/manifold/wheels/manifold3d*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
