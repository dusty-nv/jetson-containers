#!/usr/bin/env bash
set -ex

QUADRANTS_REPO="https://github.com/Genesis-Embodied-AI/quadrants"
QUADRANTS_DIR="/opt/quadrants"

git clone --branch=v${QUADRANTS_VERSION} --recursive ${quadrants_REPO} ${QUADRANTS_DIR} || \
    git clone --recursive ${quadrants_REPO} ${QUADRANTS_DIR}

# Navigate to the quadrants repository directory
cd ${QUADRANTS_DIR}

# Apply the inline assembly fix for ARM64 CUDA (if needed)
sed -i 's/"l"(value)/"r"(value)/g' quadrants/runtime/llvm/runtime_module/runtime.cpp || true
sed -i 's/match\.any\.sync\.b64  %0/match\.any\.sync\.b64  %w0/g; s/, %1/, %w1/g; s/, %2/, %w2/g' \
    quadrants/runtime/llvm/runtime_module/runtime.cpp || true


# Set environment variables for the build
export MAX_JOBS=$(nproc)
export CC=clang
export CXX=clang++
export QUADRANTS_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON"
export CUDA_VERSION=13.1
export LLVM_VERSION=20
export LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}

uv pip install "cmake<4"
# Build quadrants
./build.py
# Check if the build succeeded
if [ $? -ne 0 ]; then
    echo "❌ quadrants build failed. Exiting..."
    exit 1
fi

# List the generated wheel file(s)
ls dist/*.whl || echo "⚠️ No wheel files found!"

# Install the built quadrants package
uv pip install /opt/quadrants/dist/*.whl

# CPU BACKEND MUST BE FIXED
# python3 -c "import quadrants as ti; ti.init(arch=ti.cpu); print('✅ quadrants installed successfully!')"

# Ensure numpy is installed
uv pip install numpy

# Check if CUDA is available
if ! python3 -c "import quadrants as ti; ti.init(arch=ti.cuda)"; then
    echo "⚠️ Warning: quadrants failed to initialize CUDA!"
fi

# Check if Vulkan is available
if ! vulkaninfo > /dev/null 2>&1; then
    echo "⚠️ Warning: Vulkan is not installed or not working properly!"
fi

# Upload to PyPI if possible
twine upload --verbose /opt/quadrants/dist/*.whl || echo "⚠️ Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
