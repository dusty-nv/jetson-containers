#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MAMBA_VERSION} --depth=1 --recursive https://github.com/Adamdad/kat /opt/kat || \
git clone --depth=1 --recursive https://github.com/Adamdad/kat /opt/kat

# Navigate to the directory containing kat setup.py
cd /opt/kat

git clone --depth=1 --recursive https://github.com/johnnynunez/rational_kat_cu.git /opt/rational_kat_cu
cd /opt/rational_kat_cu

uv pip install --upgrade pip setuptools wheel twine
sed -i "/^[[:space:]]*cmdclass=/i\    setup_requires=['torch>=2.2']," setup.py
cat setup.py
uv build --wheel --out-dir /opt/rational_kat_cu/wheels .
# uv pip install /opt/rational_kat_cu/wheels/*.whl
uv pip install -e .
uv pip install timm

# Optionally upload to a repository using Twine
twine upload --verbose /opt/rational_kat_cu/wheels/*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"

cd /opt/kat
