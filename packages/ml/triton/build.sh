#!/usr/bin/env bash
# triton
set -ex

echo "============ Building triton ${TRITON_VERSION} (branch=${TRITON_BRANCH}) ============"

# Install build dependencies
apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

uv pip uninstall triton

git clone --branch ${TRITON_BRANCH} --depth=1 --recursive https://github.com/triton-lang/triton /opt/triton ||
git clone --depth=1 --recursive https://github.com/triton-lang/triton /opt/triton
cd /opt/triton

#git checkout ${TRITON_BRANCH}
#git -C third_party submodule update --init nvidia || git submodule update --init --recursive

sed -i \
    -e 's|LLVMAMDGPUCodeGen||g' \
    -e 's|LLVMAMDGPUAsmParser||g' \
    -e 's|-Werror|-Wno-error|g' \
    CMakeLists.txt

sed -i 's|^download_and_copy_ptxas|#&|' python/setup.py || :

mkdir -p third_party/cuda
ln -sf /usr/local/cuda/bin/ptxas $(pwd)/third_party/cuda/ptxas

# Ensure CUDA headers are visible to the compiler
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"

export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
export TRITON_CUOBJDUMP_PATH="${CUDA_HOME}/bin/cuobjdump"
export TRITON_NVDISASM_PATH="${CUDA_HOME}/bin/nvdisasm"
export TRITON_CUDACRT_PATH="${CUDA_HOME}/include"
export TRITON_CUDART_PATH="${CUDA_HOME}/include"
export TRITON_CUPTI_INCLUDE_PATH="${CUDA_HOME}/extras/CUPTI/include"
export TRITON_CUPTI_LIB_PATH="${CUDA_HOME}/extras/CUPTI/lib64"

uv pip install setuptools wheel pybind11 cmake ninja

uv build --no-build-isolation --wheel --out-dir /opt .

cd /
rm -rf /opt/triton

uv pip install /opt/triton*.whl

uv pip show triton
python3 -c 'import triton'
twine upload --verbose /opt/triton*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
