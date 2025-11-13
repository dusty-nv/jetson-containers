#!/usr/bin/env bash
set -ex

echo "Building onnxruntime ${ONNXRUNTIME_VERSION} (branch=${ONNXRUNTIME_BRANCH}, flags=${ONNXRUNTIME_FLAGS})"

# Detect TensorRT installation path
# Priority: Tegra location, then standard location
TENSORRT_HOME=""
CUDNN_HOME=""

if [ -f "/usr/local/cuda/targets/$(uname -m)-linux/lib/libnvinfer.so" ]; then
    # Tegra/Jetson platform - TensorRT from tar.gz
    TENSORRT_HOME="/usr/local/cuda/targets/$(uname -m)-linux/lib"
    CUDNN_HOME="/usr/local/cuda/targets/$(uname -m)-linux/lib"
    echo "Found TensorRT in Tegra location: ${TENSORRT_HOME}"
elif [ -f "/usr/lib/$(uname -m)-linux-gnu/libnvinfer.so" ]; then
    # Standard location - TensorRT from .deb packages
    TENSORRT_HOME="/usr/lib/$(uname -m)-linux-gnu"
    CUDNN_HOME="/usr/lib/$(uname -m)-linux-gnu"
    echo "Found TensorRT in standard location: ${TENSORRT_HOME}"
else
    echo "ERROR: TensorRT core library (libnvinfer.so) not found"
    echo "Searched locations:"
    echo "  - /usr/local/cuda/targets/$(uname -m)-linux/lib/libnvinfer.so"
    echo "  - /usr/lib/$(uname -m)-linux-gnu/libnvinfer.so"
    echo ""
    echo "Available CUDA libraries:"
    find /usr/local/cuda -name "libnvinfer.so*" 2>/dev/null || echo "  None found"
    echo ""
    echo "Available system libraries:"
    find /usr/lib -name "libnvinfer.so*" 2>/dev/null || echo "  None found"
    exit 1
fi

# Ensure TensorRT libraries are in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${TENSORRT_HOME}:$LD_LIBRARY_PATH

echo "Using TensorRT home: ${TENSORRT_HOME}"
echo "Using cuDNN home: ${CUDNN_HOME}"

# Memory-adaptive build parallelism to prevent OOM during CUDA kernel compilation
# CUDA compilation (especially cicc) can use 5-10GB per parallel job
detect_cuda_max_jobs() {
    local total_ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    local cicc_mem_per_job=6  # GB per cicc process (conservative estimate)
    local system_reserve=8     # GB to reserve for system/Docker/base build

    # Calculate safe MAX_JOBS based on total RAM
    local safe_jobs=$(( (total_ram_gb - system_reserve) / cicc_mem_per_job ))

    # Clamp between reasonable bounds
    if [ $safe_jobs -lt 2 ]; then safe_jobs=2; fi
    if [ $safe_jobs -gt 12 ]; then safe_jobs=12; fi  # Cap at 12 even for large systems

    echo $safe_jobs
}

ARCH=$(uname -m)
if [[ "${ARCH}" = "aarch64" ]] || [[ "${ARCH}" = "arm64" ]]; then
    export MAX_JOBS=$(detect_cuda_max_jobs)
    export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
    export MAKEFLAGS="-j${MAX_JOBS}"

    total_ram=$(free -h | awk '/^Mem:/{print $2}')
    echo ""
    echo "Memory-adaptive CUDA build configuration (ARM64):"
    echo "  Total RAM: $total_ram"
    echo "  MAX_JOBS: $MAX_JOBS (auto-detected)"
    echo "  Expected peak memory: ~$((MAX_JOBS * 6 + 8))GB (${MAX_JOBS} CUDA jobs × 6GB + 8GB overhead)"
    echo ""
else
    # x86_64 - use sensible defaults
    export MAX_JOBS=8
    export CMAKE_BUILD_PARALLEL_LEVEL=8
    export MAKEFLAGS="-j8"
    echo "Using default parallelism: MAX_JOBS=8 (x86_64)"
fi

uv pip uninstall onnxruntime || echo "onnxruntime was not previously installed"

git clone https://github.com/microsoft/onnxruntime /opt/onnxruntime
cd /opt/onnxruntime

git checkout ${ONNXRUNTIME_BRANCH} || echo "Branch ${ONNXRUNTIME_BRANCH} not found, staying on main branch"
git submodule update --init --recursive

install_dir="/opt/onnxruntime/install"

./build.sh --config Release --update --parallel --build --build_wheel --build_shared_lib \
        --skip_tests --skip_submodule_sync ${ONNXRUNTIME_FLAGS} \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable -I/usr/local/cuda/include" \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        --cmake_extra_defines CMAKE_INSTALL_PREFIX=${install_dir} \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cuda_home /usr/local/cuda --cudnn_home ${CUDNN_HOME} \
        --use_tensorrt --tensorrt_home ${TENSORRT_HOME}

cd build/Linux/Release
make install

ls -ll dist
cp dist/onnxruntime*.whl /opt
cd /

uv pip install /opt/onnxruntime*.whl
python3 -c 'import onnxruntime; print(onnxruntime.__version__);'

twine upload --verbose /opt/onnxruntime*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
tarpack upload onnxruntime-gpu-${ONNXRUNTIME_VERSION} ${install_dir} || echo "failed to upload tarball"

cd ${install_dir}
cp -r * /usr/local/
ls
#rm -rf /tmp/onnxruntime
