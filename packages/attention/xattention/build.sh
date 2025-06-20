#!/usr/bin/env bash
set -ex

echo "Building xattention ${XATTENTION_VERSION}"

git clone --depth=1 --branch=v${XATTENTION_VERSION} https://github.com/mit-han-lab/x-attention /opt/xattention ||
git clone --depth=1 https://github.com/mit-han-lab/x-attention /opt/xattention

cd /opt/xattention

sed -i 's/==/>=/g' requirements.txt
pip3 install packaging setuptools wheel
pip3 install --ignore-installed blinker


export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
pip3 wheel --wheel-dir=/opt/xattention/wheels --verbose .
# pip3 install /opt/xattention/wheels/xattn*.whl
pip3 install -e .
twine upload --verbose /opt/xattention/wheels/xattn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
