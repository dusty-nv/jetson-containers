#!/usr/bin/env bash
set -ex

echo "Installing TensorRT-Edge-LLM ${TENSORRT_EDGELLM_VERSION}"

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    git
rm -rf /var/lib/apt/lists/*
apt-get clean

git clone --branch=${TENSORRT_EDGELLM_BRANCH} --depth=1 --recurse-submodules \
    https://github.com/NVIDIA/TensorRT-Edge-LLM.git ${SOURCE_DIR} || \
git clone --depth=1 --recurse-submodules \
    https://github.com/NVIDIA/TensorRT-Edge-LLM.git ${SOURCE_DIR}

cd ${SOURCE_DIR}

apt-get update
apt-get install -y --no-install-recommends python3-dev
rm -rf /var/lib/apt/lists/*
apt-get clean

sed -i 's|torch~=2.9.1|torch>=2.9.1|g' requirements.txt
sed -i 's|"torch~=2.9.1"|"torch>=2.9.1"|g' pyproject.toml
sed -i 's|torch~=|torch>=|g' requirements.txt pyproject.toml

uv pip install .

tensorrt-edgellm-export-llm --help
tensorrt-edgellm-quantize-llm --help

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing C++ build of TensorRT-Edge-LLM ${TENSORRT_EDGELLM_VERSION}"
    exit 1
fi

/tmp/tensorrt_edgellm/build.sh

touch /tmp/tensorrt_edgellm/.done
