#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MAMBA_VERSION} --depth=1 --recursive https://github.com/johnnynunez/mamba /opt/mamba || \
git clone --depth=1 --recursive https://github.com/johnnynunez/mamba /opt/mamba

# Generate the diff dynamically
python3 /tmp/mamba/generate_diff.py
git apply /tmp/mamba/patch.diff
git diff
git status

sed -i '/torch/d' pyproject.toml
sed -i '/triton/d' pyproject.toml

pip3 install lm-eval
pip3 install --upgrade setuptools einops
pip3 install triton
export MAX_JOBS=$(nproc)
export MAMBA_FORCE_BUILD=TRUE
export MAMBA_SKIP_CUDA_BUILD=FALSE
pip3 wheel --no-build-isolation --wheel-dir=/opt/mamba/wheels . --verbose
pip3 install /opt/mamba/wheels/mamba_ssm*.whl

cd /opt/mamba

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mamba/wheels/mamba_ssm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
