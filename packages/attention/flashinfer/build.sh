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
  echo "¡Error! No existe $VERSION_FILE" >&2
  exit 1
fi
sed -i "1,\$c\\${FLASHINFER_VERSION}" version.txt
sed -i 's|options={.*| |g' setup.py
sed -i 's/"cuda-python<=12\.9"/"cuda-python<=13.1"/' setup.py
echo "Patched $REPO_DIR/setup.py"
cat setup.py

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

python3 -m pip install --no-cache-dir build setuptools wheel ninja mpi4py nvidia-ml-py einops $NVSHMEM_PKG requests

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}
export FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST

uv pip install apache-tvm-ffi
python3 -m flashinfer.aot
python3 -m build --no-isolation --wheel
# Install AOT wheel
python3 -m pip install dist/flashinfer_python-*.whl

twine upload --verbose dist/flashinfer_python-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
