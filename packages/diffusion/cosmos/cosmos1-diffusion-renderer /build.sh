#!/usr/bin/env bash
set -ex

echo "Building cosmos1-diffusion-renderer ${COSMOS_DIFF_RENDER_VERSION}"

git clone --branch=v${COSMOS_DIFF_RENDER_VERSION} --depth=1 --recursive https://github.com/nv-tlabs/cosmos1-diffusion-renderer /opt/cosmos1-diffusion-renderer || \
git clone --recursive https://github.com/nv-tlabs/cosmos1-diffusion-renderer /opt/cosmos1-diffusion-renderer

cd /opt/cosmos1-diffusion-renderer
sed -i '/decord==0.6.0/d' requirements.txt
sed -i 's/==/>=/g' requirements.txt
pip3 install decord2
pip3 install -r requirements.txt
cosmos1-diffusion-renderer_MORE_DETAILS=1 MAX_JOBS=$(nproc) \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/cosmos1-diffusion-renderer/wheels/

ls /opt/cosmos1-diffusion-renderer/wheels/
cd /

pip3 install /opt/cosmos1-diffusion-renderer/wheels/nvidia-cosmos*.whl

twine upload --verbose /opt/nvidia-cosmos*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
