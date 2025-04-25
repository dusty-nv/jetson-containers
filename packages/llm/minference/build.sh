#!/usr/bin/env bash
set -ex

pip3 install compressed-tensors decord

REPO_URL="https://github.com/microsoft/MInference"
REPO_DIR="/opt/minference"

echo "Building minference ${MINFERENCE_VERSION}"

git clone --recursive --depth=1 --branch=v${MINFERENCE_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR

cd $REPO_DIR

# export MAX_JOBS="$(nproc)" this breaks with actual flash-attention
export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

cd $REPO_DIR

pip3 wheel '.[all]' --wheel-dir $PIP_WHEEL_DIR
pip3 install $PIP_WHEEL_DIR/minference-*.whl

cd /opt/minference
pip3 show minference
python3 -c 'import minference'

twine upload --verbose $PIP_WHEEL_DIR/minference-*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"