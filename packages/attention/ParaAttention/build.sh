#!/usr/bin/env bash
set -ex

echo "Building ParaAttention ${PARAATENTTION_VERSION}"

git clone --depth=1 --branch=v${ParaAttention_VERSION} https://github.com/chengzeyi/ParaAttention /opt/paraattention ||
git clone --depth=1 https://github.com/chengzeyi/ParaAttention /opt/paraattention

cd /opt/paraattention

uv pip install packaging
uv pip install --reinstall blinker


export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

uv pip install 'setuptools>=64' 'setuptools_scm>=8'
MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/paraattention/wheels

ls /opt/paraattention/wheels
cd /

uv pip install /opt/paraattention/wheels/para_attn*.whl

twine upload --verbose /opt/paraattention/wheels/para_attn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
