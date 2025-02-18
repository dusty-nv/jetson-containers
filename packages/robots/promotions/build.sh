#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${PROTOMOTIONS_VERSION} --depth=1 --recursive https://github.com/NVlabs/ProtoMotions /opt/protomotions  || \
git clone --depth=1 --recursive https://github.com/NVlabs/ProtoMotions  /opt/protomotions

# Navigate to the directory containing mamba's setup.py
cd /opt/protomotions
git lfs fetch --all
sed -i '/torch/d; /open3d/d' requirements_genesis.txt

pip3 wheel --wheel-dir=/opt/protomotions/wheels/ --no-deps --verbose .
pip3 install --no-cache-dir --verbose /opt/protomotions/wheels/protomotions*.whl

cd /opt/protomotions

# Optionally upload to a repository using Twine
twine upload --verbose /opt/protomotions/wheels/promotions*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
