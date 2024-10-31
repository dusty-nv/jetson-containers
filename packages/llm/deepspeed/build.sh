#!/usr/bin/env bash
set -ex

echo "Building DeepSpeed ${DEEPSPEED_VERSION} (branch=${DEEPSPEED_BRANCH})"

git clone --branch=${DEEPSPEED_BRANCH} --depth=1 --recursive https://github.com/microsoft/DeepSpeed /opt/DeepSpeed
cd /opt/DeepSpeed

python3 setup.py build_ext -j$(nproc) bdist_wheel --dist-dir /opt/wheels

pip3 install --no-cache-dir --verbose /opt/wheels/deepspeed-*.whl

twine upload --verbose /opt/wheels/deepspeed-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
