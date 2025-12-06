#!/usr/bin/env bash
set -ex

TAICHI_REPO="https://github.com/Genesis-Embodied-AI/gstaichi"
TAICHI_DIR="/opt/gstaichi"

git clone --branch=v${GSTAICHI_VERSION} --recursive ${GSTAICHI_REPO} ${GSTAICHI_DIR} || \
    git clone --recursive ${GSTAICHI_REPO} ${GSTAICHI_DIR}

# Navigate to the Taichi repository directory
cd ${GSTAICHI_DIR}

# Apply the inline assembly fix for ARM64 CUDA (if needed)
sed -i 's/"l"(value)/"r"(value)/g' gstaichi/runtime/llvm/runtime_module/runtime.cpp || true
sed -i 's/match\.any\.sync\.b64  %0/match\.any\.sync\.b64  %w0/g; s/, %1/, %w1/g; s/, %2/, %w2/g' \
    gstaichi/runtime/llvm/runtime_module/runtime.cpp || true


# Set environment variables for the build
export MAX_JOBS=$(nproc)
export CC=clang
export CXX=clang++
export GSTAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON"
export CUDA_VERSION=13.1
export LLVM_VERSION=20
export LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}

uv pip install "cmake<4"
# Build GSTaichi
./build.py
# Check if the build succeeded
if [ $? -ne 0 ]; then
    echo "❌ GSTaichi build failed. Exiting..."
    exit 1
fi

# List the generated wheel file(s)
ls dist/*.whl || echo "⚠️ No wheel files found!"

# Install the built Taichi package
uv pip install /opt/gstaichi/dist/*.whl

# CPU BACKEND MUST BE FIXED
# python3 -c "import gstaichi as ti; ti.init(arch=ti.cpu); print('✅ gstaichi installed successfully!')"

# Ensure numpy is installed
uv pip install numpy

# Check if CUDA is available
if ! python3 -c "import gstaichi as ti; ti.init(arch=ti.cuda)"; then
    echo "⚠️ Warning: gstaichi failed to initialize CUDA!"
fi

# Check if Vulkan is available
if ! vulkaninfo > /dev/null 2>&1; then
    echo "⚠️ Warning: Vulkan is not installed or not working properly!"
fi

# Upload to PyPI if possible
twine upload --verbose /opt/gstaichi/dist/*.whl || echo "⚠️ Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
