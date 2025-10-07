#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${CASUALCONV1D_VERSION} --depth=1 --recursive https://github.com/Dao-AILab/causal-conv1d /opt/causalconv1d  || \
git clone --depth=1 --recursive https://github.com/Dao-AILab/causal-conv1d  /opt/causalconv1d

# Navigate to the directory containing mamba's setup.py
cd /opt/causalconv1d

# Generate the diff dynamically
python3 /tmp/causalconv1d/generate_diff.py
git apply /tmp/causalconv1d/patch.diff
git diff
git status

MAX_JOBS="$(nproc)" \
CAUSAL_CONV1D_FORCE_BUILD="TRUE" \
CAUSAL_CONV1D_SKIP_CUDA_BUILD="FALSE" \
python3 setup.py bdist_wheel --dist-dir=/opt/causalconv1d/wheels
uv pip install /opt/causalconv1d/wheels/causal_conv1d*.whl

cd /opt/causalconv1d

# Optionally upload to a repository using Twine
twine upload --verbose /opt/causalconv1d/wheels/causal_conv1d*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
