#!/usr/bin/env bash
set -ex

# Install dependencies of XGrammar
pip3 install "pybind11[global]" pre-commit

# Clone the repository if it doesn't exist
git clone --branch=v${XGRAMMAR_VERSION} --recursive --depth=1 https://github.com/mlc-ai/xgrammar /opt/xgrammar ||
git clone --recursive --depth=1 https://github.com/mlc-ai/xgrammar /opt/xgrammar

# Build and install
cd /opt/xgrammar
pre-commit install
mkdir build
cd build
cmake .. -G Ninja
ninja

# Create the wheel
cd /opt/xgrammar
pip3 wheel . --wheel-dir /opt/xgrammar/wheels/ --verbose

# Install the wheel
# Warning: version number is 0.1.5 even if actual version is 0.1.8, or 0.1.9 due to version.py not being adapted yet: https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/version.py
pip3 install /opt/xgrammar/wheels/xgrammar*.whl

# Clone the repository if it doesn't exist
git clone --branch=v${VLLM_VERSION} --recursive --depth=1 https://github.com/vllm-project/vllm /opt/vllm || 
git clone --recursive --depth=1 https://github.com/vllm-project/vllm /opt/vllm
cd /opt/vllm

env
cp /tmp/vllm/${VLLM_VERSION}.fa.diff /tmp/vllm/fa.diff
git apply /tmp/vllm/${VLLM_VERSION}.diff
git diff
git status

export MAX_JOBS=$(nproc) # this is for AGX (max 4 working on Orin NX)
export USE_CUDNN=1
export VERBOSE=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"

python3 use_existing_torch.py || echo "skipping vllm/use_existing_torch.py"

pip3 install -r requirements-build.txt -v
python3 -m setuptools_scm
pip3 wheel --no-build-isolation -v --wheel-dir=/opt/vllm/wheels .
pip3 install --no-cache-dir --verbose /opt/vllm/wheels/vllm*.whl

cd /opt/vllm
pip3 install compressed-tensors

# Optionally upload to a repository using Twine
twine upload --verbose /opt/vllm/wheels/vllm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/xgrammar/wheels/xgrammar*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
