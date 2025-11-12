#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${TRANSFORMER_ENGINE_VERSION} --depth=1 --recursive https://github.com/NVIDIA/TransformerEngine /opt/transformer_engine  || \
git clone --depth=1 --recursive https://github.com/NVIDIA/TransformerEngine /opt/transformer_engine

# Navigate to the directory containing mamba's setup.py
cd /opt/transformer_engine

# git apply /tmp/TRANSFORMER_ENGINE/patch.diff
# git diff
# git status
uv pip install nvidia-mathdx
uv pip install --upgrade pip setuptools wheel pybind11[global] scikit-build cmake ninja
export NVTE_FRAMEWORK=pytorch

if [[ "${TORCH_CUDA_ARCH_LIST}" == "8.7" ]]; then
  export NVTE_CUDA_ARCHS="8.7"
else
  export NVTE_CUDA_ARCHS="87;89;90a;100a;103a;110a;120a;121a"
fi

MAX_JOBS=$(nproc) \
NVTE_FRAMEWORK=pytorch \
NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS \
uv build --wheel --no-build-isolation . --out-dir /opt/transformer_engine/wheels --verbose
uv pip install /opt/transformer_engine/wheels/transformer_engine*.whl

cd /opt/transformer_engine



# Optionally upload to a repository using Twine
twine upload --verbose /opt/transformer_engine/wheels/transformer_engine*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
