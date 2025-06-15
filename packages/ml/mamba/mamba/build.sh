#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MAMBA_VERSION} --depth=1 --recursive https://github.com/johnnynunez/mamba /opt/mamba  || \
git clone --depth=1 --recursive https://github.com/johnnynunez/mamba  /opt/mamba

# Navigate to the directory containing mamba's setup.py
cd /opt/mamba

# Generate the diff dynamically
python3 /tmp/mamba/generate_diff.py
git apply /tmp/mamba/patch.diff
git diff
git status

sed -i '/torch/d' pyproject.toml
sed -i '/triton/d' pyproject.toml

MAX_JOBS=16 \
MAMBA_FORCE_BUILD="TRUE" \
MAMBA_SKIP_CUDA_BUILD="FALSE" \
python3 setup.py bdist_wheel --dist-dir=/opt/mamba/wheels
pip3 install /opt/mamba/wheels/mamba*.whl

cd /opt/mamba

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mamba/wheels/mamba*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
