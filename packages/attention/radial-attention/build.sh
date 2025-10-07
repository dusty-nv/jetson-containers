#!/usr/bin/env bash
set -ex

echo "Building RadialAttention ${RADIAL_ATTENTION_VERSION}"

git clone --depth=1 --branch=v${RADIAL_ATTENTION_VERSION} https://github.com/mit-han-lab/radial-attention /opt/radial-attention ||
git clone --depth=1 https://github.com/mit-han-lab/radial-attention /opt/radial-attention

cd /opt/radial-attention
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
sed -i 's/==/>=/g' requirements.txt
uv pip install -U -r requirements.txt
