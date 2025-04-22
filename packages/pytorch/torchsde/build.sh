#!/usr/bin/env bash
set -ex
echo "Building torchsde ${TORCHSDE_VERSION}"

BRANCH_VERSION=$(echo "$torchsde_VERSION" | sed 's/^\([0-9]*\.[0-9]*\)\.0$/\1/')
git clone --branch=v${torchsde_VERSION} --recursive --depth=1 https://github.com/pytorch/vision /opt/torchsde ||
git clone --branch=release/${BRANCH_VERSION} --recursive --depth=1 https://github.com/pytorch/vision /opt/torchsde ||
git clone --recursive --depth=1 https://github.com/pytorch/vision /opt/torchsde
cd /opt/torchsde

BUILD_VERSION=${torchsde_VERSION} \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/torchsde

pip3 install /opt/torchsde*.whl
pip3 show torchsde && python3 -c 'import torchsde; print(torchsde.__version__);'

twine upload --verbose /opt/torchsde*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
