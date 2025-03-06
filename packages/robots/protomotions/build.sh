#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${PROTOMOTIONS_VERSION} --depth=1 --recursive https://github.com/NVlabs/ProtoMotions /opt/protomotions  || \
git clone --depth=1 --recursive https://github.com/NVlabs/ProtoMotions  /opt/protomotions

pip3 install --upgrade setuptools wheel wandb

# Navigate to the directory containing mamba's setup.py
cd /opt/protomotions
git lfs fetch --all
sed -i '/torch/d; /open3d/d; /wandb/d; /PyOpenGL==3\.1\.4/d' requirements_genesis.txt


pip3 install open3d PyOpenGL PyOpenGL_accelerate
pip3 install -e .
pip3 install -r requirements_genesis.txt
pip3 install -e isaac_utils
pip3 install -e poselib

pip3 wheel --wheel-dir=/opt/protomotions/wheels/ --verbose .
# pip3 install /opt/protomotions/wheels/protomotions*.whl

cd /opt/protomotions

# Optionally upload to a repository using Twine
twine upload --verbose /opt/protomotions/wheels/protomotions*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
