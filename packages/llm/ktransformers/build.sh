#!/usr/bin/env bash
set -ex

echo "Building ktransformers version ${KTRANSFORMERS_VERSION}"
# Clone the repository if it doesn't exist
git clone --branch=v${ktransformers_VERSION} --recursive --depth=1 https://github.com/kvcache-ai/ktransformers /opt/ktransformers || 
git clone --recursive --depth=1 https://github.com/kvcache-ai/ktransformers /opt/ktransformers
cd /opt/ktransformers

pip3 wheel --no-build-isolation -v --wheel-dir=/opt/ktransformers/wheels .
pip3 install /opt/ktransformers/wheels/ktransformers*.whl

cd /opt/ktransformers
pip3 install compressed-tensors

# Optionally upload to a repository using Twine
twine upload --verbose /opt/ktransformers/wheels/ktransformers*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
