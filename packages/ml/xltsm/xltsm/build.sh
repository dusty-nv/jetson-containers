#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${XLSTM_VERSION} --depth=1 --recursive https://github.com/NX-AI/xlstm /opt/xlstm  || \
git clone --depth=1 --recursive https://github.com/NX-AI/xlstm  /opt/xlstm

# Navigate to the directory containing xlstm's setup.py
cd /opt/xlstm

pip3 wheel . -v --no-deps -w /opt/xlstm/wheels/
pip3 install /opt/xlstm/wheels/xlstm*.whl

cd /opt/xlstm

# Optionally upload to a repository using Twine
twine upload --verbose /opt/xlstm/wheels/xlstm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
