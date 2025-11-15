#!/usr/bin/env bash
set -ex

echo "Building FlashInfer ${FLASHINFER_VERSION}"

REPO_URL="https://github.com/flashinfer-ai/flashinfer"
REPO_DIR="/opt/flashinfer"

git clone --recursive --depth=1 --branch=v${FLASHINFER_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR

cd $REPO_DIR
VERSION_FILE="version.txt"
if [[ ! -f "$VERSION_FILE" ]]; then
  echo "¡Error! Not exists $VERSION_FILE" >&2
  exit 1
fi
sed -i "1,\$c\\${FLASHINFER_VERSION}" version.txt
# Detect CUDA major version
CUDA_VER=$(/usr/local/cuda/bin/nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+)\..*/\1/')
echo "Detected CUDA version: $CUDA_VER"

# Choose NVSHMEM package name based on CUDA version
if [[ "$CUDA_VER" -ge 13 ]]; then
    NVSHMEM_PKG="nvidia-nvshmem-cu13"
elif [[ "$CUDA_VER" -ge 12 ]]; then
    NVSHMEM_PKG="nvidia-nvshmem-cu12"
else
    echo "Unsupported CUDA version: $CUDA_VER" >&2
    exit 1
fi
echo "Using NVSHMEM package: $NVSHMEM_PKG"

python3 -m pip install --no-cache-dir build setuptools wheel ninja mpi4py nvidia-ml-py einops $NVSHMEM_PKG requests tqdm

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}
if [[ "${TORCH_CUDA_ARCH_LIST}" == "8.7" ]]; then
    export FLASHINFER_CUDA_ARCH_LIST="8.7"
    uv pip install cuda-python==12.8.0 --index-url https://pypi.org/simple/
else
  export FLASHINFER_CUDA_ARCH_LIST="8.7 9.0a 10.0a 10.3a 11.0f 12.0f"
fi

uv pip install apache-tvm-ffi
sed -i '/nvidia-cutlass-dsl>=4\.2\.1/d' requirements.txt
uv pip install -r requirements.txt --prerelease=allow
cd /opt/flashinfer/
uv build --no-build-isolation -v --wheel . --out-dir /opt/wheels/ || echo "failed to build flashinfer wheel"
cd /opt/flashinfer/flashinfer-cubin
uv build --no-build-isolation -v --wheel . --out-dir /opt/wheels/ || echo "failed to build flashinfer-cubin wheel"
cd /opt/flashinfer/flashinfer-jit-cache
uv build --no-build-isolation -v --wheel . --out-dir /opt/wheels/ || echo "failed to build flashinfer-jit-cache wheel"
# Install AOT wheel
python3 -m pip install /opt/wheels/flashinfer*.whl

twine upload --verbose /opt/wheels/flashinfer*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
