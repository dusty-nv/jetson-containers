#!/usr/bin/env bash
set -ex

echo "Building blocksparseattention ${BLOCKSPARSEATTN_VERSION}"

# Memory optimization: Limit parallel jobs to prevent OOM
# Use min(4, nproc/2) to balance speed vs memory usage
TOTAL_CORES=$(nproc)
if [ $TOTAL_CORES -gt 8 ]; then
    MAX_JOBS=4
else
    MAX_JOBS=$((TOTAL_CORES / 2))
    if [ $MAX_JOBS -lt 1 ]; then
        MAX_JOBS=1
    fi
fi

echo "Memory-optimized build: Using $MAX_JOBS parallel jobs (total cores: $TOTAL_CORES)"

git clone --depth=1 --branch=v${BLOCKSPARSEATTN_VERSION} https://github.com/mit-han-lab/Block-Sparse-Attention /opt/block_sparse_attn ||
git clone --depth=1 https://github.com/mit-han-lab/Block-Sparse-Attention /opt/block_sparse_attn

cd /opt/block_sparse_attn

# Clean up git history to save memory
rm -rf .git

uv pip install packaging setuptools wheel
uv pip install --reinstall blinker

# Set memory-optimized build parameters
export MAX_JOBS=$MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=-O3 -DCMAKE_C_FLAGS_RELEASE=-O3"

echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

# Build with memory optimization
MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
python3 setup.py bdist_wheel --dist-dir /opt/block_sparse_attn/wheels

# Clean up build artifacts to free memory
find . -name "*.o" -delete
find . -name "*.so" -not -path "./wheels/*" -delete
find . -name "build" -type d -exec rm -rf {} + 2>/dev/null || true

ls /opt/block_sparse_attn/wheels
cd /

uv pip install /opt/block_sparse_attn/wheels/block_sparse_attn*.whl

twine upload /opt/block_sparse_attn/wheels/block_sparse_attn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
