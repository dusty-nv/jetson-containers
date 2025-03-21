#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${3DGRUT_VERSION} --depth=1 --recursive https://github.com/nv-tlabs/3dgrut /opt/3DGRUT || \
git clone --depth=1 --recursive https://github.com/nv-tlabs/3dgrut /opt/3DGRUT

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/3DGRUT


# Set GCC-11 and G++-11 as the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

MAX_JOBS=$(nproc) \
pip3 wheel . -w /opt/3DGRUT/wheels --verbose

pip3 install /opt/3DGRUT/wheels/3DGRUT*.whl

cd /opt/3DGRUT

# Optionally upload to a repository using Twine
twine upload --verbose /opt/3DGRUT/wheels/3DGRUT*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
