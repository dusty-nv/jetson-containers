#!/usr/bin/env bash
set -ex

echo "Building MemPalace ${MEMPALACE_VERSION}"

REPO_URL="https://github.com/milla-jovovich/mempalace.git"

if [ "$MEMPALACE_VERSION" = "latest" ]; then
    git clone --depth 1 ${REPO_URL} /opt/mempalace
else
    git clone --depth 1 --branch v${MEMPALACE_VERSION} ${REPO_URL} /opt/mempalace
fi

cd /opt/mempalace

sed -i 's/==/>=/g' requirements.txt

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

uv build --wheel . -v --no-build-isolation --out-dir /opt/mempalace/wheels/

ls /opt/mempalace/wheels
cd /

uv pip install /opt/mempalace/wheels/mempalace*.whl

twine upload --verbose /opt/mempalace/wheels/mempalace*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

mempalace --help

echo "MemPalace installed successfully"
