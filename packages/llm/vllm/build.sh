#!/usr/bin/env bash
set -ex

pip3 install pre-commit nanobind==2.5.0
# Clone the repository if it doesn't exist
git clone --branch=${VLLM_BRANCH} --recursive --depth=1 https://github.com/vllm-project/vllm /opt/vllm ||
git clone --recursive --depth=1 https://github.com/vllm-project/vllm /opt/vllm

cd /opt/vllm
env

# cp /tmp/vllm/${VLLM_VERSION}.fa.diff /tmp/vllm/fa.diff
# git apply /tmp/vllm/${VLLM_VERSION}.diff

echo "Applying vLLM CMake patches…"
git apply -p1 /tmp/vllm/0.10.2.diff || echo "patch already applied"

# File "/opt/venv/lib/python3.12/site-packages/gguf/gguf_reader.py"
# `newbyteorder` was removed from the ndarray class in NumPy 2.0
sed -i \
  -e 's|^gguf.*|gguf|g' \
  -e 's|^opencv-python-headless.*||g' \
  -e 's|^mistral_common.*|mistral_common|g' \
  requirements/common.txt

grep gguf requirements/common.txt


export USE_CUDNN=1
export VERBOSE=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"
export DG_JIT_USE_NVRTC=1 # DeepGEMM now supports NVRTC with up to 10x compilation speedup

python3 use_existing_torch.py || echo "skipping vllm/use_existing_torch.py"

pip3 install -r requirements/build.txt -v
python3 -m setuptools_scm

ARCH=$(uname -i)
if [ "${ARCH}" = "aarch64" ]; then
      export NVCC_THREADS=1
      export CUDA_NVCC_FLAGS="-Xcudafe --threads=1"
      export MAKEFLAGS='-j2'
      export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
      export NINJAFLAGS='-j2'
fi

pip3 wheel --no-build-isolation -v --wheel-dir=/opt/vllm/wheels .
pip3 install /opt/vllm/wheels/vllm*.whl

cd /opt/vllm
pip3 install compressed-tensors

# Optionally upload to a repository using Twine
twine upload --verbose /opt/vllm/wheels/vllm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
