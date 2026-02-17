#!/usr/bin/env bash
set -ex

echo "Building cosmos1-diffusion-renderer ${COSMOS_DIFF_RENDER_VERSION}"

git clone --branch=v${COSMOS_DIFF_RENDER_VERSION} --depth=1 --recursive https://github.com/nv-tlabs/cosmos1-diffusion-renderer /opt/cosmos1-diffusion-renderer || \
git clone --recursive https://github.com/nv-tlabs/cosmos1-diffusion-renderer /opt/cosmos1-diffusion-renderer

cd /opt/cosmos1-diffusion-renderer
sed -i '/decord==0.6.0/d' requirements.txt
sed -i 's/==/>=/g' requirements.txt
uv pip install decord2
uv pip install -r requirements.txt
export MAX_JOBS=$(nproc)
uv build --wheel --no-deps --verbose . --out-dir /opt/cosmos1-diffusion-renderer/wheels/

ls /opt/cosmos1-diffusion-renderer/wheels/
cd /

uv pip install /opt/cosmos1-diffusion-renderer/wheels/nvidia_cosmos*.whl

twine upload --verbose /opt/nvidia_cosmos*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
