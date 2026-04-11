#!/usr/bin/env bash
set -ex

echo "Building multi-turboquant ${MULTI_TURBOQUANT_VERSION}"

git clone --depth=1 --branch=v${MULTI_TURBOQUANT_VERSION} https://github.com/rookiemann/multi-turboquant /opt/multi-turboquant ||
git clone --depth=1 https://github.com/rookiemann/multi-turboquant /opt/multi-turboquant

cd /opt/multi-turboquant

sed -i 's/==/>=/g' requirements*.txt 2>/dev/null || true

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
uv build --wheel . -v --no-build-isolation --out-dir /opt/multi-turboquant/wheels/

ls /opt/multi-turboquant/wheels
cd /

uv pip install /opt/multi-turboquant/wheels/multi_turboquant*.whl

twine upload --verbose /opt/multi-turboquant/wheels/multi_turboquant*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
