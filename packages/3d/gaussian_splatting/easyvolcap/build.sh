#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${EASYVOLCAP_VERSION} --depth=1 --recursive https://github.com/zju3dv/EasyVolcap /opt/easyvolcap || \
git clone --depth=1 --recursive https://github.com/zju3dv/EasyVolcap /opt/easyvolcap

cd /opt/easyvolcap
export MAX_JOBS=$(nproc)

# Build python wheel from source
uv pip install -U -r requirements.txt
uv pip install PyOpenGL pdbr tqdm
uv build --wheel . --out-dir $PIP_WHEEL_DIR --verbose
uv pip install $PIP_WHEEL_DIR/easyvolcap*.whl

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/easyvolcap*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
