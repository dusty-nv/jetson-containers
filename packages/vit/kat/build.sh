#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MAMBA_VERSION} --depth=1 --recursive https://github.com/Adamdad/kat /opt/kat || \
git clone --depth=1 --recursive https://github.com/Adamdad/kat /opt/kat

# Navigate to the directory containing kat setup.py
cd /opt/kat

git clone https://github.com/Adamdad/rational_kat_cu.git /opt/rational_kat_cu
cd /opt/rational_kat_cu

pip3 wheel --no-build-isolation --wheel-dir=/opt/rational_kat_cu/wheels .
pip3 install --no-cache-dir --verbose /opt/rational_kat_cu/wheels/*.whl
pip install -e . --no-cache-dir
pip install timm==1.0.3

# Optionally upload to a repository using Twine
twine upload --verbose /opt/rational_kat_cu/wheels/*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"

cd /opt/kat