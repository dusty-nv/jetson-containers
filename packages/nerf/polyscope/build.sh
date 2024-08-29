#!/usr/bin/env bash
set -ex

echo "Building polyscope ${POLYSCOPE_VERSION}"

git clone --branch=v${POLYSCOPE_VERSION} --depth=1 --recursive https://github.com/nmwsharp/polyscope-py /opt/polyscope

cd /opt/polyscope

MAX_JOBS=$(nproc) \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

ls /opt
cd /

pip3 install --no-cache-dir --verbose /opt/wheels/polyscope*.whl

twine upload --verbose /opt/wheels/polyscope*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"