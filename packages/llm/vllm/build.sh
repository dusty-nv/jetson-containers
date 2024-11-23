#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${VLLM_VERSION} --recursive --depth=1 https://github.com/vllm-project/vllm /opt/vllm || 
git clone --recursive --depth=1 https://github.com/vllm-project/vllm /opt/vllm
cd /opt/vllm

# apply patches
git apply /tmp/vllm/${VLLM_VERSION}.diff
git diff

export MAX_JOBS=$(nproc)
export USE_CUDNN=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"

python3 use_existing_torch.py || echo "skipping vllm/use_existing_torch.py"

pip3 install -r requirements-build.txt
python3 -m setuptools_scm
pip3 wheel --no-build-isolation --verbose --wheel-dir=/opt/vllm/wheels .
pip3 install --no-cache-dir --verbose /opt/vllm/wheels/vllm*.whl

cd /opt/vllm
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/vllm/wheels/vllm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
