#!/usr/bin/env bash
set -ex

TAICHI_REPO="https://github.com/johnnynunez/taichi"
TAICHI_DIR="/opt/taichi"

git clone --branch=v${TAICHI_VERSION} --recursive ${TAICHI_REPO} ${TAICHI_DIR} || \
    git clone --recursive ${TAICHI_REPO} ${TAICHI_DIR}

# Navigate to the Taichi repository directory
cd ${TAICHI_DIR}

# Apply the inline assembly fix for ARM64 CUDA (if needed)
sed -i 's/"l"(value)/"r"(value)/g' taichi/runtime/llvm/runtime_module/runtime.cpp || true
sed -i 's/match\.any\.sync\.b64  %0/match\.any\.sync\.b64  %w0/g; s/, %1/, %w1/g; s/, %2/, %w2/g' \
    taichi/runtime/llvm/runtime_module/runtime.cpp || true


# Set environment variables for the build
export MAX_JOBS=$(nproc)
export CC=clang
export CXX=clang++
export TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON"
export CUDA_VERSION=12.8
export LLVM_VERSION=20
export LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}

pip3 install "cmake<4"
# Build Taichi
./build.py
# Check if the build succeeded
if [ $? -ne 0 ]; then
    echo "❌ Taichi build failed. Exiting..."
    exit 1
fi

# List the generated wheel file(s)
ls dist/*.whl || echo "⚠️ No wheel files found!"

# Install the built Taichi package
pip3 install /opt/taichi/dist/*.whl

# CPU BACKEND MUST BE FIXED
# python3 -c "import taichi as ti; ti.init(arch=ti.cpu); print('✅ Taichi installed successfully!')"

# Ensure numpy is installed
pip3 install numpy

# Check if CUDA is available
if ! python3 -c "import taichi as ti; ti.init(arch=ti.cuda)"; then
    echo "⚠️ Warning: Taichi failed to initialize CUDA!"
fi

# Check if Vulkan is available
if ! vulkaninfo > /dev/null 2>&1; then
    echo "⚠️ Warning: Vulkan is not installed or not working properly!"
fi

# Upload to PyPI if possible
twine upload --verbose /opt/taichi/dist/*.whl || echo "⚠️ Failed to upload wheel to ${TWINE_REPOSITORY_URL}"