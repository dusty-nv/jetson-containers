#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MAMBA_VERSION} --depth=1 --recursive https://github.com/Adamdad/kat /opt/kat || \
git clone --depth=1 --recursive https://github.com/Adamdad/kat /opt/kat

# Navigate to the directory containing kat setup.py
cd /opt/kat

git clone https://github.com/Adamdad/rational_kat_cu.git
cd rational_kat_cu

pip3 wheel --no-build-isolation --wheel-dir=/opt/kat/wheels .
pip3 install --no-cache-dir --verbose /opt/kat/wheels/*.whl


cd /opt/kat
pip install timm==1.0.3
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/kat/wheels/*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
