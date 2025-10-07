#!/usr/bin/env bash
set -ex
echo "Building torchtext ${TORCHTEXT_VERSION}"

BRANCH_VERSION=$(echo "TORCHTEXT_VERSION" | sed 's/^\([0-9]*\.[0-9]*\)\.0$/\1/')
git clone --branch=v${TORCHTEXT_VERSION} --recursive --depth=1 https://github.com/pytorch/text /opt/torchtext ||
git clone --branch=release/${BRANCH_VERSION} --recursive --depth=1 https://github.com/pytorch/text /opt/torchtext ||
git clone --recursive --depth=1 https://github.com/pytorch/text /opt/torchtext
cd /opt/torchtext

BUILD_VERSION=${TORCHTEXT_VERSION} \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/torchtext

uv pip install /opt/torchtext*.whl
uv pip show torchtext && python3 -c 'import torchtext; print(torchtext.__version__);'

twine upload --verbose /opt/torchtext*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
