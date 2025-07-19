#!/usr/bin/env bash
set -ex

echo "Building CuPy ${CUPY_VERSION}"

git clone --branch ${CUPY_VERSION} --depth 1 --recursive https://github.com/cupy/cupy /opt/cupy
cd /opt/cupy

pip3 install fastrlock
pip3 wheel --no-build-isolation --wheel-dir /opt/cupy/wheels/ .
cp /opt/cupy/wheels/*.whl /opt

pip3 install /opt/cupy/wheels/*.whl
pip3 show cupy && python3 -c 'import cupy; print(cupy.show_config())'

twine upload --verbose /opt/cupy/wheels/*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
