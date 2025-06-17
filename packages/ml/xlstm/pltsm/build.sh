#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${PLSTM_VERSION} --depth=1 --recursive https://github.com/ml-jku/plstm /opt/plstm  || \
git clone --depth=1 --recursive https://github.com/ml-jku/plstm  /opt/plstm

# Navigate to the directory containing plstm's setup.py
cd /opt/plstm

pip3 wheel . -v --no-deps -w /opt/plstm/wheels/
pip3 install /opt/plstm/wheels/plstm*.whl

cd /opt/plstm

# Optionally upload to a repository using Twine
twine upload --verbose /opt/plstm/wheels/plstm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
