#!/usr/bin/env bash
set -ex
echo "Building torchvision ${TORCHVISION_VERSION}"

BRANCH_VERSION=$(echo "$TORCHVISION_VERSION" | sed 's/^\([0-9]*\.[0-9]*\)\.0$/\1/')
git clone --branch=v${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision /opt/torchvision ||
git clone --branch=release/${BRANCH_VERSION} --recursive --depth=1 https://github.com/pytorch/vision /opt/torchvision ||
git clone --recursive --depth=1 https://github.com/pytorch/vision /opt/torchvision
cd /opt/torchvision

BUILD_VERSION=${TORCHVISION_VERSION} \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/torchvision

uv pip install /opt/torchvision*.whl
uv pip show torchvision && python3 -c 'import torchvision; print(torchvision.__version__);'

twine upload --verbose /opt/torchvision*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
