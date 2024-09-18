#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${FLAX_VERSION} --depth=1 --recursive https://github.com/google/flax /opt/flax || \
git clone --depth=1 --recursive https://github.com/google/flax /opt/flax

cd /opt/flax

pip3 wheel --no-build-isolation --wheel-dir=/opt/flax/wheels .
pip3 install --no-cache-dir --verbose /opt/flax/wheels/flax*.whl

cd /opt/flax
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/flax/wheels/flax*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
