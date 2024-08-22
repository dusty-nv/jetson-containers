#!/usr/bin/env bash
set -ex

echo "Building nerfstudio ${NERFSTUDIO_VERSION}"

git clone --branch=v${NERFSTUDIO_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/nerfstudio /opt/nerfstudio

cd /opt/pymeshlab

MAX_JOBS=$(nproc) \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

ls /opt
cd /

pip3 install --no-cache-dir --verbose /opt/nerfstudio*.whl

twine upload --verbose /opt/nerfstudio*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"