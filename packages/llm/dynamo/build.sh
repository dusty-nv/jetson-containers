#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${DYNAMO_VERSION} --depth=1 --recursive https://github.com/ai-dynamo/dynamo /opt/dynamo || \
git clone --depth=1 --recursive https://github.com/ai-dynamo/dynamo /opt/dynamo

# Navigate to the directory containing dynamo's setup.py
cd /opt/dynamo

export MAX_JOBS=$(nproc)
pip3 wheel --no-build-isolation --wheel-dir=/opt/dynamo/wheels . --verbose
pip3 install /opt/dynamo/wheels/ai-dynamo*.whl

cd /opt/dynamo

# Optionally upload to a repository using Twine
twine upload --verbose /opt/dynamo/wheels/ai-dynamo*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
