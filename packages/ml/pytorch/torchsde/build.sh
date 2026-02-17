#!/usr/bin/env bash
set -ex
echo "Building torchsde ${TORCHSDE_VERSION}"

BRANCH_VERSION=$(echo "$TORCH_SDE_VERSION" | sed 's/^\([0-9]*\.[0-9]*\)\.0$/\1/')
git clone --branch=v${TORCH_SDE_VERSION} --recursive --depth=1 https://github.com/google-research/torchsde /opt/torchsde ||
git clone --branch=release/${BRANCH_VERSION} --recursive --depth=1 https://github.com/google-research/torchsde /opt/torchsde ||
git clone --recursive --depth=1 https://github.com/google-research/torchsde /opt/torchsde
cd /opt/torchsde

BUILD_VERSION=${TORCH_SDE_VERSION} \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/torchsde/wheels

cd ../
uv pip install /opt/torchsde/wheels/torchsde*.whl
uv pip show torchsde && python3 -c 'import torchsde; print(torchsde.__version__);'

twine upload --verbose /opt/torchsde/wheels/torchsde*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
