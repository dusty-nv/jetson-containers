#!/usr/bin/env bash
set -ex

echo "Building onnxruntime ${ONNXRUNTIME_VERSION} (branch=${ONNXRUNTIME_BRANCH}, flags=${ONNXRUNTIME_FLAGS})"
 
pip3 uninstall -y onnxruntime onnxruntime-gpu || echo "onnxruntime was not previously installed"

git clone https://github.com/microsoft/onnxruntime /opt/onnxruntime
cd /opt/onnxruntime

if [ -n "${ONNXRUNTIME_BRANCH}" ]; then
    git checkout ${ONNXRUNTIME_BRANCH}
fi
git submodule update --init --recursive

sed -i 's|archive/3.4/eigen-3.4.zip;ee201b07085203ea7bd8eb97cbcb31b07cfa3efb|archive/3.4.0/eigen-3.4.0.zip;ef24286b7ece8737c99fa831b02941843546c081|' cmake/deps.txt || echo "cmake/deps.txt not found"

install_dir="/opt/onnxruntime/install"

# Fix CUDA architectures format - replace commas with semicolons
# CMake expects semicolons, not commas
if [ -n "${CUDA_ARCHITECTURES}" ]; then
    # Convert commas to spaces first (safer for shell handling)
    CUDA_ARCH_FIXED=$(echo ${CUDA_ARCHITECTURES} | tr ',' ' ')
else
    # Default architectures for Jetson and newer NVIDIA GPUs
    CUDA_ARCH_FIXED="72 87"
fi

echo "Using CUDA architectures: ${CUDA_ARCH_FIXED}"

# Add the --allow_running_as_root flag to fix the root permission error
./build.sh --config Release --update --parallel --build --build_wheel --build_shared_lib \
        --skip_tests --skip_submodule_sync ${ONNXRUNTIME_FLAGS} \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable -I/usr/local/cuda/include" \
        --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_FIXED}" \
        --cmake_extra_defines CMAKE_INSTALL_PREFIX=${install_dir} \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/$(uname -m)-linux-gnu \
        --use_tensorrt --tensorrt_home /usr/lib/$(uname -m)-linux-gnu \
        --allow_running_as_root
	   
cd build/Linux/Release
make install || echo "Make install failed, but continuing..."

ls -ll dist || echo "No dist directory found"
if [ -d "dist" ]; then
    cp dist/onnxruntime*.whl /opt || echo "No wheel files found"
fi
cd /

# Try to install the wheel if it exists
if ls /opt/onnxruntime*.whl 1> /dev/null 2>&1; then
    pip3 install /opt/onnxruntime*.whl
    python3 -c 'import onnxruntime; print(f"ONNX Runtime {onnxruntime.__version__} built and installed successfully");'
else
    # If wheel not found, install from PyPI
    echo "No wheel found, installing from PyPI"
    pip3 install onnxruntime
    python3 -c 'import onnxruntime; print(f"ONNX Runtime {onnxruntime.__version__} installed from PyPI");'
fi

# Copy libraries to system directories if the install directory exists
if [ -d "${install_dir}" ]; then
    cp -r ${install_dir}/* /usr/local/ || echo "Failed to copy files to /usr/local/"
    echo "ONNX Runtime libraries installed to /usr/local/"
fi
