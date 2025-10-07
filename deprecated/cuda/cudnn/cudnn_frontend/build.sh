#!/usr/bin/env bash
set -ex

echo "Building cudnn_frontend ${cudnn_frontend_VERSION}"

REPO_URL="https://github.com/NVIDIA/cudnn-frontend"
REPO_DIR="/opt/cudnn_frontend"

git clone --recursive --depth=1 --branch=v${cudnn_frontend_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}


cd $REPO_DIR
uv pip install -U -r requirements.txt
uv build --wheel --out-dir $REPO_DIR --verbose .
uv pip install /opt/cudnn_frontend/nvidia_cudnn_frontend-*.whl

twine upload --verbose /opt/cudnn_frontend/nvidia_cudnn_frontend-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
