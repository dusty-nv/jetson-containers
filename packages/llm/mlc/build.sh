#!/usr/bin/env bash
set -ex

echo "Building MLC ${MLC_VERSION} (commit=${MLC_COMMIT}) against external TVM at ${TVM_HOME:-/opt/tvm}"

# dependencies needed by MLC build
apt-get update
apt-get install -y --no-install-recommends libzstd-dev ccache ninja-build
rm -rf /var/lib/apt/lists/*
apt-get clean

# clone the sources
git clone --recursive https://github.com/mlc-ai/mlc-llm ${SOURCE_DIR}
cd ${SOURCE_DIR}
git fetch origin ${MLC_COMMIT} || true
git checkout ${MLC_COMMIT}
git submodule update --init --recursive

# apply optional patch to the source
if [ -s /tmp/mlc/patch.diff ]; then
	git apply /tmp/mlc/patch.diff || echo "patch did not apply; continuing"
fi

git status
git diff --submodule=diff || true

# add extras to the source
cp /tmp/mlc/benchmark.py ${SOURCE_DIR}/ || true

# Prepare build configuration (CMake config.cmake)
mkdir -p build
cd build
touch config.cmake

# Determine the target CUDA arch (fallback to 110)
archs=${CUDAARCHS:-${CUDA_ARCHITECTURES}}
target_arch=$(echo "$archs" | tr ';' '\n' | tail -n1)

{
	echo "set(TVM_SOURCE_DIR ${TVM_HOME:-/opt/tvm})";
	echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)";
	echo "set(USE_CUDA ON)";
	echo "set(USE_CUTLASS ON)";
	echo "set(USE_CUBLAS ON)";
	echo "set(USE_ROCM OFF)";
	echo "set(USE_VULKAN OFF)";
	echo "set(USE_METAL OFF)";
	echo "set(USE_OPENCL OFF)";
	echo "set(USE_OPENCL_ENABLE_HOST_PTR OFF)";
	echo "set(USE_THRUST ON)";
	echo "set(CMAKE_CUDA_ARCHITECTURES ${target_arch})";
	echo "set(USE_FLASHINFER ON)";
	echo "set(FLASHINFER_ENABLE_FP4_E2M1 ON)";
	echo "set(FLASHINFER_ENABLE_BF16 ON)";
	echo "set(FLASHINFER_ENABLE_F16 ON)";
	echo "set(FLASHINFER_ENABLE_FP8_E4M3 ON)";
	echo "set(FLASHINFER_ENABLE_FP8_E5M2 ON)";
	echo "set(FLASHINFER_ENABLE_FP8_E8M0 ON)";
	echo "set(FLASHINFER_GEN_GROUP_SIZES 1 4 6 8)";
	echo "set(FLASHINFER_GEN_PAGE_SIZES 16)";
	echo "set(FLASHINFER_GEN_HEAD_DIMS 128)";
	echo "set(FLASHINFER_GEN_KV_LAYOUTS 0 1)";
	echo "set(FLASHINFER_GEN_POS_ENCODING_MODES 0 1)";
	echo "set(FLASHINFER_GEN_ALLOW_FP16_QK_REDUCTIONS \"false\")";
	echo "set(FLASHINFER_GEN_CASUALS \"false\" \"true\")";
	echo "set(FLASHINFER_CUDA_ARCHITECTURES ${target_arch})";
} >> config.cmake

# Configure and build
cmake ..
make -j$(nproc)

# Build and install mlc-llm python package
cd ${SOURCE_DIR}
if [ -f setup.py ]; then
	python3 setup.py --verbose bdist_wheel --dist-dir /opt || true
fi

cd python
python3 setup.py --verbose bdist_wheel --dist-dir /opt

uv pip install /opt/mlc*.whl

ln -sf ${TVM_HOME:-/opt/tvm}/3rdparty "$(uv pip show tvm | awk '/Location:/ {print $2}')/tvm/3rdparty" || true

# Upload wheels (best-effort)
twine upload --verbose /opt/mlc_llm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/mlc_chat*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
