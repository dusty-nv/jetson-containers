#!/usr/bin/env bash
set -ex

pip3 install pre-commit nanobind wheel setuptools twine
# Clone the repository if it doesn't exist
git clone --branch=${UNSLOTH_VERSION} --recursive --depth=1 https://github.com/unslothai/unsloth /opt/unsloth ||
git clone --recursive --depth=1 https://github.com/unslothai/unsloth /opt/unsloth

cd /opt/unsloth

export MAX_JOBS=$(nproc) # this is for AGX (max 4 working on Orin NX)

pip3 wheel --no-build-isolation -v --wheel-dir=/opt/unsloth/wheels .
pip3 install /opt/unsloth/wheels/unsloth*.whl

sed -i 's/==/>=/g' pyproject.toml && \
sed -i 's/~=/>=/g' pyproject.toml

cd /opt/unsloth
pip3 install compressed-tensors unsloth_zoo cut_cross_entropy

# Optionally upload to a repository using Twine
twine upload --verbose /opt/unsloth/wheels/unsloth*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
