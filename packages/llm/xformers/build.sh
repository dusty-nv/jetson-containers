#!/usr/bin/env bash
set -ex

echo "Building xformers ${XFORMERS_VERSION}"

git clone --branch=v${XFORMERS_VERSION} --depth=1 --recursive https://github.com/facebookresearch/xformers /opt/xformers ||
git clone --depth=1 --recursive https://github.com/facebookresearch/xformers /opt/xformers

cd /opt/xformers

XFORMERS_MORE_DETAILS=1 MAX_JOBS=$(nproc) \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

ls /opt
cd /

pip3 install --no-cache-dir --verbose /opt/xformers*.whl

twine upload --verbose /opt/xformers*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
