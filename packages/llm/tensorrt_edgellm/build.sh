#!/usr/bin/env bash
set -ex

echo "Building TensorRT-Edge-LLM C++ runtime ${TENSORRT_EDGELLM_VERSION}"

cd ${SOURCE_DIR}

TRT_PACKAGE_DIR="/usr"

if [ ! -d "${TRT_PACKAGE_DIR}/include" ] || [ ! -d "${TRT_PACKAGE_DIR}/lib" ]; then
    echo "TensorRT not found at ${TRT_PACKAGE_DIR}, searching..."
    TRT_DIR=$(find /usr/local -maxdepth 1 -name "TensorRT*" -type d | head -1)
    if [ -n "$TRT_DIR" ]; then
        TRT_PACKAGE_DIR="$TRT_DIR"
    fi
fi

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}"

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    if [ -f cmake/aarch64_linux_toolchain.cmake ]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake"
    fi
fi

mkdir -p build
cd build

cmake .. ${CMAKE_ARGS}
make -j$(nproc)

echo "TensorRT-Edge-LLM C++ build complete"
ls -la examples/llm/ 2>/dev/null || true
ls -la examples/multimodal/ 2>/dev/null || true
