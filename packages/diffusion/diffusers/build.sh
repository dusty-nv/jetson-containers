#!/usr/bin/env bash
set -ex

echo "Building diffusers ${DIFFUSERS_VERSION}"

git clone --branch=v${DIFFUSERS_VERSION} --depth=1 --recursive https://github.com/huggingface/diffusers /opt/diffusers

cd /opt/diffusers

DIFFUSERS_MORE_DETAILS=1 MAX_JOBS=$(nproc) \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

ls /opt
cd /

pip3 install --no-cache-dir --verbose /opt/diffusers*.whl

twine upload --verbose /opt/diffusers*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
