#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MLSTM_KERNELS_VERSION} --depth=1 --recursive https://github.com/NX-AI/mlstm_kernels /opt/mlstm_kernels  || \
git clone --depth=1 --recursive https://github.com/NX-AI/mlstm_kernels  /opt/mlstm_kernels

# Navigate to the directory containing mlstm_kernels's setup.py
cd /opt/mlstm_kernels

pip3 wheel . -v --no-deps -w /opt/mlstm_kernels/wheels/
pip3 install /opt/mlstm_kernels/wheels/mlstm_kernels*.whl

cd /opt/mlstm_kernels

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mlstm_kernels/wheels/mlstm_kernels*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
