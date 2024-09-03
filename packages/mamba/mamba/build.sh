#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MAMBA_VERSION} --depth=1 --recursive https://github.com/state-spaces/mamba /opt/mamba || \
git clone --depth=1 --recursive https://github.com/state-spaces/mamba /opt/mamba

# Navigate to the directory containing mamba's setup.py
cd /opt/mamba 
pip3 install causal-conv1d>=1.4.0
pip3 install lm-eval
pip3 install --upgrade setuptools einops
export MAMBA_FORCE_BUILD=TRUE
export MAMBA_SKIP_CUDA_BUILD=FALSE
pip wheel --no-build-isolation --verbose --wheel-dir=/wheels .

cd /opt/mamba

pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mamba/wheels/mamba_ssm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
