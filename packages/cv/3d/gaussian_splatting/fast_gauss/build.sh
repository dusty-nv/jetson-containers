#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${FAST_GAUSIAN_VERSION} --depth=1 --recursive https://github.com/dendenxu/fast-gaussian-rasterization /opt/fast_gauss || \
git clone --depth=1 --recursive https://github.com/dendenxu/fast-gaussian-rasterization /opt/fast_gauss

cd /opt/fast_gauss
export MAX_JOBS=$(nproc)

# Build python wheel from source
uv pip install PyOpenGL pdbr tqdm ujson ruamel.yaml
uv build --wheel . --out-dir $PIP_WHEEL_DIR --verbose
uv pip install $PIP_WHEEL_DIR/fast_gauss*.whl

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/fast_gauss*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
