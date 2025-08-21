#!/usr/bin/env bash
set -uo pipefail
set -x

# Expected: PYTORCH_BUILD_VERSION is something like "2.9.0" (requested base version)
: "${PYTORCH_BUILD_VERSION:?Set PYTORCH_BUILD_VERSION, e.g. 2.9.0}"

REPO_URL="${REPO_URL:-https://github.com/pytorch/pytorch}"
REPO_DIR="/opt/pytorch"

ORIGINAL_REQUEST="${PYTORCH_BUILD_VERSION}"

echo "Requested PyTorch base version: ${PYTORCH_BUILD_VERSION}"
echo "Repo URL: ${REPO_URL}"

# --- Clone: use cached repo if available, otherwise clone fresh ---
if [[ -d "/opt/pytorch-cache" ]]; then
    echo "Using cached PyTorch repository, refreshing..."
    cd /opt/pytorch-cache

    # Fetch latest changes and tags
    git fetch --tags --force --prune origin

    # Try to checkout the requested version first
    if git tag -l "v${PYTORCH_BUILD_VERSION}" | grep -q "v${PYTORCH_BUILD_VERSION}"; then
        echo "Found tag v${PYTORCH_BUILD_VERSION}, checking it out..."
        git checkout "v${PYTORCH_BUILD_VERSION}"
        git submodule update --init --recursive --force
        echo "Successfully checked out v${PYTORCH_BUILD_VERSION}"
    else
        echo "Tag v${PYTORCH_BUILD_VERSION} not found, using main branch"
        git reset --hard origin/main
        git submodule update --init --recursive --force
        echo "Using main branch (latest)"
    fi

    echo "Repository refreshed from cache"

    # Copy refreshed repo to expected location
    echo "Copying refreshed repository to /opt/pytorch..."
    cd /
    rm -rf /opt/pytorch  # Remove any existing directory
    cp -r /opt/pytorch-cache /opt/pytorch
    cd /opt/pytorch
    echo "Repository copied to /opt/pytorch"

else
    echo "No cached repository found, cloning fresh..."
    # Fallback to original cloning logic
    if git clone --branch "v${PYTORCH_BUILD_VERSION}" --depth=1 --recursive "${REPO_URL}" "${REPO_DIR}"; then
        echo "Cloned exact tag v${PYTORCH_BUILD_VERSION}"
    else
        echo "Tag v${PYTORCH_BUILD_VERSION} not found; cloning default branch"
        git clone --depth=1 --recursive "${REPO_URL}" "${REPO_DIR}"
        cd "${REPO_DIR}"
        git fetch --tags --force --prune
    fi
    cd "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Helper: parse semver tag list safely (vX.Y.Z only)
latest_release_tag() {
  git tag --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -n 1
}

# Is HEAD exactly at a release tag?
HEAD_TAG="$(git tag --points-at HEAD | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n1 || true)"

DATE_UTC="$(date -u +%Y%m%d)"
GIT_SHA_SHORT="$(git rev-parse --short=7 HEAD)"

# CUDA detection -> cuXYZ or cpu
detect_cuda_tag() {
  # Prefer explicit $CUDA_VERSION
  if [[ -n "${CUDA_VERSION:-}" ]]; then
    ver="${CUDA_VERSION}"
  else
    # Try nvcc
    if command -v nvcc >/dev/null 2>&1; then
      # nvcc --version | tail -n1 | awk '{print $6}'  -> e.g., V13.0.1
      vline="$(nvcc --version | grep -Eo 'V[0-9]+\.[0-9]+' | head -n1 || true)"
      ver="${vline#V}"
    elif [[ -f "/usr/local/cuda/version.json" ]]; then
      ver="$(python3 - <<'PY'
import json,sys
try:
  with open('/usr/local/cuda/version.json') as f:
    d=json.load(f)
  v=d.get('cuda',{}).get('version') or d.get('version')
  print(v or '')
except Exception:
  print('')
PY
)"
    elif [[ -f "/usr/local/cuda/version.txt" ]]; then
      # e.g., "CUDA Version 13.0.0"
      ver="$(grep -Eo '[0-9]+\.[0-9]+' /usr/local/cuda/version.txt | head -n1 || true)"
    else
      ver=""
    fi
  fi

  if [[ -z "${ver}" ]]; then
    echo "cpu"
    return
  fi

  # Normalize: 13.0 -> cu130, 12.6 -> cu126
  maj="$(echo "${ver}" | cut -d. -f1)"
  min="$(echo "${ver}" | cut -d. -f2)"
  if [[ -z "${maj}" || -z "${min}" ]]; then
    echo "cpu"
  else
    # Fix: Use %d%d instead of %d%02d to avoid padding with zeros
    printf "cu%d%d" "${maj}" "${min}"
  fi
}

CUDA_TAG="$(detect_cuda_tag)"
echo "CUDA tag detected: ${CUDA_TAG}"

# Compute EFFECTIVE_VERSION
if [[ -n "${HEAD_TAG}" && "${HEAD_TAG}" == "v${PYTORCH_BUILD_VERSION}" ]]; then
  # Exact release build
  EFFECTIVE_VERSION="${PYTORCH_BUILD_VERSION}"
  echo "Building exact release from ${HEAD_TAG}"
else
  LAST_TAG="$(latest_release_tag || true)"
  if [[ -z "${LAST_TAG}" ]]; then
    echo "WARNING: No release tags found; defaulting base to 0.0.0"
    BASE_MAJ=0; BASE_MIN=0; BASE_PAT=0
  else
    BASE="${LAST_TAG#v}"
    IFS='.' read -r BASE_MAJ BASE_MIN BASE_PAT <<<"${BASE}"
  fi

  NEXT_MINOR="${BASE_MAJ:-0}.$(( ${BASE_MIN:-0} + 1 )).0"

  # PEP 440 dev + local metadata (cu tag and short SHA as segments)
  EFFECTIVE_VERSION="${NEXT_MINOR}.dev${DATE_UTC}+${CUDA_TAG}.g${GIT_SHA_SHORT}"

  echo "HEAD is not at v${PYTORCH_BUILD_VERSION}; using dev snapshot"
  echo "Last tag: ${LAST_TAG:-<none>}; Next minor: ${NEXT_MINOR}"
fi

export PYTORCH_BUILD_VERSION="${EFFECTIVE_VERSION}"
export PYTORCH_BUILD_NUMBER="${PYTORCH_BUILD_NUMBER:-1}"

# Build info
echo "=== PyTorch Build Information ==="
echo "Requested base:  ${ORIGINAL_REQUEST}"
echo "HEAD tag:        ${HEAD_TAG:-<none>}"
echo "Actual commit:   $(git rev-parse HEAD)"
echo "Commit date:     $(git log -1 --format=%cd)"
echo "Commit message:  $(git log -1 --pretty=format:'%s')"
echo "CUDA tag:        ${CUDA_TAG}"
echo "=================================="

# -------- Optional cpuinfo patch (kept from your original) --------
CPUINFO_PATCH=third_party/cpuinfo/src/arm/linux/aarch64-isa.c || true
if [[ -f "${CPUINFO_PATCH}" ]]; then
  sed -i 's|cpuinfo_log_error|cpuinfo_log_warning|' "${CPUINFO_PATCH}" || true
  grep 'PR_SVE_GET_VL' "${CPUINFO_PATCH}" || echo "patched ${CPUINFO_PATCH}" || true
  tail -20 "${CPUINFO_PATCH}" || true
fi

# Fix cuSPARSELt detection by creating symbolic links
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

# NCCL environment safety
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}

# Set parallel jobs specific number
export MAX_JOBS=$(( $(nproc) - 1 ))
export CMAKE_BUILD_PARALLEL_LEVEL=$(( $(nproc) - 1 ))

# Build deps
pip3 install -r requirements.txt
pip3 install scikit-build ninja
pip3 install 'cmake<4'
export USE_PRIORITIZED_TEXT_FOR_LD=1

# NVCC fatbin compression tuning (optional)
if [[ "${CUDA_TAG}" =~ ^cu ]]; then
  CUDA_MAJOR="${CUDA_TAG#cu}"
  CUDA_MAJOR="${CUDA_MAJOR:0:1}"
  if [[ "${CUDA_MAJOR}" == "1" ]]; then
    # crude: if CUDA 12.6 specifically is desired, allow override via env
    if [[ "${CUDA_VERSION:-}" == "12.6" ]]; then
      export TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=balance"
    else
      export TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=size"
    fi
  fi
fi

# --- Resource monitoring during build ---
echo "=== Pre-build resource status ==="
echo "Memory: $(free -h | grep Mem)"
echo "Disk space: $(df -h /opt)"
echo "CPU cores: $(nproc)"
echo "System load: $(uptime)"
echo "=================================="

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

# Feature toggles (keep your defaults)
export USE_CUDNN=1
export USE_CUSPARSELT=1
export USE_CUDSS=1
export USE_CUFILE=1
export USE_NATIVE_ARCH=1
export USE_DISTRIBUTED=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_TENSORRT=0
export USE_BLAS="${USE_BLAS:-}"
export BLAS="${BLAS:-}"

# Disable testing to avoid compiler crashes
export BUILD_TESTING=OFF
export BUILD_TESTING_CPP=OFF
export CMAKE_BUILD_TESTING=OFF
export BUILD_TEST=0
export USE_GTEST=0

# If on CUDA 12, leave only SMs supported
if [[ "${CUDA_TAG}" == cu12* ]]; then
    echo "CUDA 12 detected, leaving only SMs supported (TORCH_CUDA_ARCH_LIST='8.7')"
    export TORCH_CUDA_ARCH_LIST="8.7"
elif [[ "${CUDA_TAG}" == cu13* ]]; then
    echo "CUDA 13 detected, turn off build with CUDSS (as 0.6 supported on CUDA 12)."
    export USE_CUDSS=0
else
    echo "*** NOT CUDA 12 NOR CUDA 13."
fi

# If on Thor (SBSA), leave only the SMs capable SMs
if [[ "${IS_SBSA}" == "True" ]]; then
    echo "SBSA detected, leaving only SMs capable (TORCH_CUDA_ARCH_LIST='11.0;12.1')"
    export TORCH_CUDA_ARCH_LIST="11.0;12.1"
else
    echo "*** IS_SBSA NOT detected."
fi

echo "=== Starting PyTorch build ==="
echo "Build command: python3 setup.py bdist_wheel --dist-dir /opt"

# Use a temporary file to capture both output and exit code
python3 setup.py bdist_wheel --dist-dir /opt 2>&1 | tee /tmp/pytorch_build.log

BUILD_EXIT_CODE=$?
BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))

echo "=== Build completed at $(date) ==="
echo "Build duration: ${BUILD_DURATION} seconds"
echo "Build exit code: ${BUILD_EXIT_CODE}"

# Analyze build log for common failure patterns
echo "=== Build log analysis ==="
if [[ -f /tmp/pytorch_build.log ]]; then
    echo "Last 50 lines of build output:"
    tail -50 /tmp/pytorch_build.log

    echo "Checking for common failure patterns:"
    grep -i "error\|fail\|killed\|oom\|timeout\|interrupt" /tmp/pytorch_build.log | tail -10 || echo "No error patterns found"

    echo "Checking for memory-related issues:"
    grep -i "memory\|alloc\|malloc\|mmap" /tmp/pytorch_build.log | tail -5 || echo "No memory-related messages found"

    echo "Checking for build system issues:"
    grep -i "ninja\|cmake\|make" /tmp/pytorch_build.log | tail -5 || echo "No build system messages found"
fi

# Check final resource status
echo "=== Post-build resource status ==="
echo "Memory: $(free -h | grep Mem)"
echo "Disk space: $(df -h /opt)"
echo "System load: $(uptime)"

# Exit with build result
if [[ $BUILD_EXIT_CODE -eq 0 ]]; then
    echo "=== Installing PyTorch wheel ==="
    pip3 install /opt/torch*.whl

    # Verify installation
    python3 -c 'import torch; print(f"PyTorch {torch.__version__} installed successfully")'

    # Verify installation in detail
    python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.backends.cudnn.version()); print(torch.__config__.show());'

    # Upload wheel to PyPI
    twine upload --verbose /opt/torch*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

    # Clean up source to save space
    cd /
    rm -rf /opt/pytorch

    echo "Build SUCCESSFUL"
    exit 0
else
    echo "Build FAILED with exit code $BUILD_EXIT_CODE"
    echo "Check /tmp/pytorch_build.log for detailed output"
    exit $BUILD_EXIT_CODE
fi

