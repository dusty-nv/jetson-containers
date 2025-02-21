#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
echo "Cloning genesis version ${GENESIS_VERSION}"
git clone --branch=v${GENESIS_VERSION} --depth=1 --recursive https://github.com/Genesis-Embodied-AI/Genesis.git /opt/genesis ||
git clone --depth=1 --recursive https://github.com/Genesis-Embodied-AI/Genesis.git /opt/genesis


# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/genesis
sed -i '/taichi/d; /tetgen/d' pyproject.toml
pip3 install -U tetgen
mkdir build && \
cd build && \
cmake .. -DCUDA_ENABLED=ON \
            -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} && \
make -j $(nproc) && \
make install && \
cd /opt/genesis && \
pip3 wheel . -w /opt/genesis/wheels && pip3 install /opt/genesis/wheels/genesis-*.whl

# Verify the contents of the /opt directory
ls /opt/genesis/wheels

# Return to the root directory
cd /

pip3 install --no-cache-dir --verbose /opt/genesis/wheels/genesis*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/genesis/wheels/genesis*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
