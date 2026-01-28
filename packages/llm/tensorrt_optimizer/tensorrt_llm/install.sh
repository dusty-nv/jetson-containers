#!/usr/bin/env bash
set -ex

# Patched: use system python to detect torch
PYTORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
DEBIAN_FRONTEND=noninteractive
apt-get update
REQUIREMENTS_FILENAME="requirements.txt"
DEV_REQUIREMENTS_FILENAME="requirements-dev.txt"

if [[ "${TRT_LLM_BRANCH}" == *"jetson"* ]]; then
    REQUIREMENTS_FILENAME="requirements-jetson.txt"
    DEV_REQUIREMENTS_FILENAME="requirements-dev-jetson.txt"
    apt-get install -y python3-libnvinfer
fi

rm -rf /var/lib/apt/lists/*
apt-get clean

bash ${TMP_DIR}/install_cusparselt.sh

uv pip install polygraphy mpi4py

# Patched: Install local TensorRT wheel to satisfy dependencies before requirements.txt
# This prevents uv from trying to find tensorrt on PyPI (which fails for Tegra)
uv pip install /usr/local/lib/python3.10/dist-packages/tensorrt-10.3.0-cp310-none-linux_aarch64.whl

if [ -s ${SOURCE_TAR} ]; then
        echo "extracting TensorRT-LLM sources from ${TRT_LLM_SOURCE}"
        mkdir -p ${SOURCE_DIR}
        tar -xzf ${SOURCE_TAR} -C ${SOURCE_DIR}
else
        if [ -s ${GIT_PATCHES} ]; then
                echo "applying git patches from ${TRT_LLM_PATCH}"
                git apply ${GIT_PATCHES}
        fi

    # Patched: Remove tensorrt from requirements to prevent PyPI fetch
    sed -i '/^tensorrt/d' "${REQUIREMENTS_FILENAME}"
    
    # Existing sed commands
    sed -i 's|^torch>.*|torch|' "${REQUIREMENTS_FILENAME}"
    sed -i 's|nvidia-cudnn.*||' "${REQUIREMENTS_FILENAME}"
    sed -i 's|cuda-python.*|cuda-python|' "${REQUIREMENTS_FILENAME}"
    sed -i 's|flashinfer-python.*|flashinfer-python|' "${REQUIREMENTS_FILENAME}"
    sed -i 's|typing-extensions.*|typing-extensions|' "${DEV_REQUIREMENTS_FILENAME}"

    git status
    git diff --submodule=diff
fi

if [ "$FORCE_BUILD" == "on" ]; then
        echo "Forcing build of TensorRT-LLM ${TRT_LLM_VERSION}"
        exit 1
fi

uv pip install -r ${REQUIREMENTS_FILENAME}
uv pip install tensorrt_llm==${TRT_LLM_VERSION}

uv pip uninstall torch && uv pip install torch==${PYTORCH_VERSION}
