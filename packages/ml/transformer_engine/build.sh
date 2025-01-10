#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${TRANSFORMER_ENGINE_VERSION} --depth=1 --recursive https://github.com/Dao-AILab/causal-conv1d /opt/transformer_engine  || \
git clone --depth=1 --recursive https://github.com/Dao-AILab/causal-conv1d  /opt/transformer_engine

# Navigate to the directory containing mamba's setup.py
cd /opt/transformer_engine  

git apply /tmp/CASUALCONV1D/patch.diff
git diff
git status

MAX_JOBS=$(nproc) \
CAUSAL_CONV1D_FORCE_BUILD="TRUE" \
CAUSAL_CONV1D_SKIP_CUDA_BUILD="FALSE" \
python3 setup.py bdist_wheel --dist-dir=/opt/transformer_engine/wheels
pip3 install --no-cache-dir --verbose /opt/transformer_engine/wheels/causal_conv1d*.whl

cd /opt/transformer_engine

pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/transformer_engine/wheels/causal_conv1d*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
