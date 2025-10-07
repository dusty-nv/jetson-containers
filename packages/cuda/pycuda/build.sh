#!/usr/bin/env bash
set -ex
echo "Building PyCUDA ${PYCUDA_VERSION}"

git clone --branch=v${PYCUDA_VERSION} --depth=1 --recursive https://github.com/inducer/pycuda /opt/pycuda || \
git clone --depth=1 --recursive https://github.com/inducer/pycuda /opt/pycuda
cd /opt/pycuda

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

python3 setup.py --verbose build_ext --inplace bdist_wheel --dist-dir /opt

cd /opt
rm -rf /opt/pycuda

uv pip install /opt/pycuda*.whl
uv pip show pycuda && python3 -c 'import pycuda; print(pycuda.VERSION_TEXT)'

twine upload --verbose /opt/pycuda*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
