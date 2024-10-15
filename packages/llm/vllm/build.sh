#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${VLLM_VERSION} --depth=1 --recursive https://github.com/vllm-project/vllm /opt/vllm || \
git clone --depth=1 --recursive https://github.com/vllm-project/vllm /opt/vllm

# Navigate to the directory containing vllm's setup.py
cd /opt/vllm

# Fetch all tags after shallow clone to ensure correct versioning
git fetch --unshallow --tags || echo "Failed to unshallow, continuing with shallow clone"

export MAX_JOBS=$(nproc)
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"

python3 use_existing_torch.py || echo "skipping vllm/use_existing_torch.py"

pip3 install -r requirements-build.txt
pip3 wheel --no-build-isolation --verbose --wheel-dir=/opt/vllm/wheels .
pip3 install --no-cache-dir --verbose /opt/vllm/wheels/vllm*.whl

cd /opt/vllm
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/vllm/wheels/vllm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
