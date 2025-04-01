#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${TRANSFORMER_ENGINE_VERSION} --depth=1 --recursive https://github.com/NVIDIA/TransformerEngine /opt/transformer_engine  || \
git clone --depth=1 --recursive https://github.com/NVIDIA/TransformerEngine /opt/transformer_engine

# Navigate to the directory containing mamba's setup.py
cd /opt/transformer_engine  

# git apply /tmp/TRANSFORMER_ENGINE/patch.diff
# git diff
# git status

MAX_JOBS=$(nproc) \
NVTE_FRAMEWORK=pytorch \
NVTE_CUDA_ARCHS=${CUDAARCHS} \
python3 setup.py bdist_wheel --dist-dir=/opt/transformer_engine/wheels
pip3 install /opt/transformer_engine/wheels/transformer_engine*.whl

cd /opt/transformer_engine



# Optionally upload to a repository using Twine
twine upload --verbose /opt/transformer_engine/wheels/transformer_engine*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
