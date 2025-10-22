#!/usr/bin/env bash
set -ex

echo "Building xattention ${XATTENTION_VERSION}"

git clone --depth=1 --branch=v${XATTENTION_VERSION} https://github.com/mit-han-lab/x-attention /opt/xattention ||
git clone --depth=1 https://github.com/mit-han-lab/x-attention /opt/xattention

cd /opt/xattention

sed -i 's/==/>=/g' requirements.txt
uv pip install packaging setuptools wheel
uv pip install --reinstall blinker


export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
uv build --wheel --out-dir /opt/xattention/wheels --verbose .
# uv pip install /opt/xattention/wheels/xattn*.whl
uv pip install -e .
twine upload --verbose /opt/xattention/wheels/xattn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
