#!/usr/bin/env bash
set -ex

TAICHI_REPO="https://github.com/johnnynunez/taichi"
TAICHI_DIR="/opt/taichi"
LLVM_VERSION=15

# Clone the repository if it doesn't exist
if [ ! -d "$TAICHI_DIR" ]; then
    git clone --branch=v${TAICHI_VERSION} --depth=1 --recursive ${TAICHI_REPO} ${TAICHI_DIR} || \
    git clone --depth=1 --recursive ${TAICHI_REPO} ${TAICHI_DIR}
fi

# Navigate to the Taichi repository directory
cd ${TAICHI_DIR}

# Apply the inline assembly fix for ARM64 CUDA (if needed)
sed -i 's/"l"(value)/"r"(value)/g' taichi/runtime/llvm/runtime_module/runtime.cpp || true
sed -i 's/match\.any\.sync\.b64  %0/match\.any\.sync\.b64  %w0/g; s/, %1/, %w1/g; s/, %2/, %w2/g' \
    taichi/runtime/llvm/runtime_module/runtime.cpp || true

# Ensure LLVM 18 is installed
wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
chmod +x /tmp/llvm.sh

# Force overwrites to prevent package conflicts
echo 'Dpkg::Options {"--force-overwrite";};' > /etc/apt/apt.conf.d/99_force_overwrite

# Install LLVM 15 with all components
# /tmp/llvm.sh ${LLVM_VERSION} all
add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-15 main"
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 15CF4D18AF4F7421
apt-get update
apt-get install llvm-15 clang-15 lldb-15 --yes


# Clean up temp config file
rm -f /etc/apt/apt.conf.d/99_force_overwrite

# Ensure LLVM binaries are linked properly
ln -sf /usr/bin/llvm-config-${LLVM_VERSION} /usr/bin/llvm-config

# Set environment variables for the build
export MAX_JOBS=$(nproc)
export TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON"
export CC=/usr/lib/llvm-${LLVM_VERSION}/bin/clang
export CXX=/usr/lib/llvm-${LLVM_VERSION}/bin/clang++
export CUDA_VERSION=12.8
export LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}

update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-15 100
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
update-alternatives --install /usr/bin/opt opt /usr/bin/opt-15 100
update-alternatives --install /usr/bin/llc llc /usr/bin/llc-15 100

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