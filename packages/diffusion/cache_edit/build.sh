#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${CACHE_DIT_VERSION} --depth=1 --recursive https://github.com/vipshop/cache-dit /opt/cache_dit  || \
git clone --depth=1 --recursive https://github.com/vipshop/cache-dit  /opt/cache_dit

# Navigate to the directory containing cache_dit's setup.py
cd /opt/cache_dit

uv build --wheel . -v --no-deps --out-dir /opt/cache_dit/wheels/
uv pip install /opt/cache_dit/wheels/cache_dit*.whl

cd /opt/cache_dit

# Optionally upload to a repository using Twine
twine upload --verbose /opt/cache_dit/wheels/cache_dit*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
