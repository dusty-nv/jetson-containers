#!/usr/bin/env bash
set -ex

echo "Building blocksparseattention ${BLOCKSPARSEATTN_VERSION}"

git clone --depth=1 --branch=v${BLOCKSPARSEATTN_VERSION} https://github.com/mit-han-lab/Block-Sparse-Attention /opt/block_sparse_attn ||
git clone --depth=1 https://github.com/mit-han-lab/Block-Sparse-Attention /opt/block_sparse_attn

cd /opt/block_sparse_attn

pip3 install packaging setuptools wheel
pip3 install --ignore-installed blinker


export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/block_sparse_attn/wheels

ls /opt/block_sparse_attn/wheels
cd /

pip3 install /opt/block_sparse_attn/wheels/block_sparse_attn*.whl

twine upload --verbose /opt/block_sparse_attn/wheels/block_sparse_attn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
