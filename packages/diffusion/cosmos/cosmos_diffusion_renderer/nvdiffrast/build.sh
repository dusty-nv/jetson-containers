#!/usr/bin/env bash
set -ex

echo "Building nvdiffrast ${NVDIFFRAST_VERSION}"

git clone --branch=v${NVDIFFRAST_VERSION} --depth=1 --recursive https://github.com/NVlabs/nvdiffrast /opt/nvdiffrast || \
git clone --recursive https://github.com/NVlabs/nvdiffrast /opt/nvdiffrast

cd /opt/nvdiffrast

pip3 install -U pip setuptools wheel
export MAX_JOBS=$(nproc)
python3 setup.py --verbose bdist_wheel --dist-dir /opt/nvdiffrast/wheels/
pip3 install -e .
pip3 install /opt/nvdiffrast/wheels/nvdiffrast*.whl

twine upload --verbose /opt/nvdiffrast/wheels/nvdiffrast*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
