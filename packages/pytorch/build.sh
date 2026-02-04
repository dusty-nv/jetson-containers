#!/usr/bin/env bash
# Python builder
set -ex

echo "Building PyTorch ${PYTORCH_BUILD_VERSION}"

# build from source
git clone --branch "v${PYTORCH_BUILD_VERSION}" --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch ||
git clone --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch
cd /opt/pytorch

# https://github.com/pytorch/pytorch/issues/138333
CPUINFO_PATCH=third_party/cpuinfo/src/arm/linux/aarch64-isa.c
sed -i 's|cpuinfo_log_error|cpuinfo_log_warning|' ${CPUINFO_PATCH}
grep 'PR_SVE_GET_VL' ${CPUINFO_PATCH} || echo "patched ${CPUINFO_PATCH}"
tail -20 ${CPUINFO_PATCH}

uv pip install -r requirements.txt
uv pip install scikit-build ninja


#TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
# https://github.com/pytorch/pytorch/pull/157791/files#diff-f271c3ed0c135590409465f4ad55c570c418d2c0509bbf1b1352ebdd1e6611d1
if [[ "$CUDA_VERSION" == "12.6" ]]; then
  export TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=balance"
else
  export TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=size"
fi

echo "=== Fixing cuSPARSELt detection ==="
if [[ -f "/usr/lib/aarch64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0.8.0.4" ]]; then
    echo "Found cuSPARSELt library, creating symbolic links..."

    # Create symbolic link in standard location
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0.8.0.4 /usr/lib/aarch64-linux-gnu/libcusparseLt.so

    # Also create versioned symlink
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0.8.0.4 /usr/lib/aarch64-linux-gnu/libcusparseLt.so.0

    # Verify the links
    ls -la /usr/lib/aarch64-linux-gnu/libcusparseLt*
    echo "cuSPARSELt symbolic links created"
else
    echo "cuSPARSELt library not found in expected location"
fi

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
if [[ "${ENABLE_DISTRIBUTED_JETSON_NCCL:-0}" == "1" ]]; then
    echo "Enabling NCCL distributed support for Jetson"
    export USE_SYSTEM_NCCL=1
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
    export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
else
    echo "NCCL distributed support for Jetson is disabled"
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
    export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
fi

# Start resource monitoring in background
monitor_resources() {
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Memory: $(free -m | grep Mem | awk '{print $3"/"$2"MB ("$3*100/$2"%)"}') - Load: $(uptime | awk -F'load average:' '{print $2}') - Disk: $(df -m /opt | tail -1 | awk '{print $3"/"$2"MB ("$3*100/$2"%)"}')"
        sleep 60
    done
}

monitor_resources &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    if [[ -n "$MONITOR_PID" ]]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

# --- Build progress tracking ---
echo "=== Starting PyTorch build at $(date) ==="
echo "Build command: python3 setup.py bdist_wheel --dist-dir /opt"
echo "Environment variables:"
env | grep -E "(USE_|BLAS|CUDA|TORCH)" | sort

# Start build with progress monitoring
BUILD_START=$(date +%s)

export USE_PRIORITIZED_TEXT_FOR_LD=1 # mandatory for ARM
export PYTORCH_BUILD_NUMBER=1
export USE_CUDNN=1
export USE_CUSPARSELT=1
export USE_CUDSS=1
export USE_CUFILE=1
export USE_XCCL=1
export USE_C10D_XCCL=1
export USE_DISTRIBUTED=1
export USE_NCCL=1
export USE_XCCL=1
export USE_NATIVE_ARCH=0
export USE_TENSORPIPE=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_TENSORRT=0
export USE_BLAS="$USE_BLAS"
export BLAS="$BLAS"
python3 setup.py bdist_wheel --dist-dir /opt

cd /
rm -rf /opt/pytorch
# install the compiled wheel
uv pip install /opt/torch*.whl
python3 -c 'import torch; print(f"PyTorch {torch.__version__} installed successfully")'

# Verify installation in detail
python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.backends.cudnn.version()); print(torch.__config__.show());'

twine upload --verbose /opt/torch*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
