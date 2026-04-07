#!/usr/bin/env bash
set -ex

echo "Building TriAttention ${TRIATTENTION_VERSION}"

git clone --depth=1 --branch=v${TRIATTENTION_VERSION} https://github.com/WeianMao/triattention /opt/triattention ||
git clone --depth=1 https://github.com/WeianMao/triattention /opt/triattention

cd /opt/triattention

sed -i 's/==/>=/g' requirements.txt

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
uv build --wheel . -v --no-build-isolation --out-dir /opt/triattention/wheels/

ls /opt/triattention/wheels
cd /

uv pip install /opt/triattention/wheels/triattention*.whl

twine upload --verbose /opt/triattention/wheels/triattention*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
