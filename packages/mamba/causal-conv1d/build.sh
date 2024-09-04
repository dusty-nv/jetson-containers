#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${CASUALCONV1D_VERSION} --depth=1 --recursive https://github.com/Dao-AILab/causal-conv1d /opt/causalconv1d  || \
git clone --depth=1 --recursive https://github.com/Dao-AILab/causal-conv1d  /opt/causalconv1d 

# Navigate to the directory containing mamba's setup.py
cd /opt/causalconv1d  

git apply /tmp/CASUALCONV1D/patch.diff
git diff
git status

pip3 wheel --no-build-isolation --wheel-dir=/opt/causalconv1d/wheels .
pip3 install --no-cache-dir --verbose /opt/causalconv1d/wheels/causal_conv1d*.whl

cd /opt/causalconv1d

pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/causalconv1d/wheels/causal_conv1d*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
