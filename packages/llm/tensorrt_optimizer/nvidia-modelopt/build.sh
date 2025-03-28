#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MAMBA_VERSION} --depth=1 --recursive https://github.com/NVIDIA/TensorRT-Model-Optimizer /opt/nvidia_modelopt || \
git clone --depth=1 --recursive https://github.com/NVIDIA/TensorRT-Model-Optimizer /opt/nvidia_modelopt

cd /opt/nvidia_modelopt

pip3 install lm-eval
pip3 install --upgrade setuptools einops
pip3 install triton
export MAX_JOBS=$(nproc)
pip3 wheel --no-build-isolation --wheel-dir=/opt/nvidia_modelopt/wheels . --verbose
pip3 install /opt/nvidia_modelopt/wheels/nvidia-modelopt*.whl

cd /opt/mamba

# Optionally upload to a repository using Twine
twine upload --verbose /opt/nvidia_modelopt/wheels/nvidia-modelopt*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
