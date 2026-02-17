#!/usr/bin/env bash
set -ex

echo "Building CuPy ${CUPY_VERSION}"

git clone --branch ${CUPY_VERSION} --depth 1 --recursive https://github.com/cupy/cupy /opt/cupy
cd /opt/cupy

uv pip install fastrlock
uv build --wheel --no-build-isolation -v --out-dir /opt/cupy/wheels/ .
cp /opt/cupy/wheels/*.whl /opt

uv pip install /opt/cupy/wheels/*.whl
uv pip show cupy
twine upload --verbose /opt/cupy/wheels/*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
