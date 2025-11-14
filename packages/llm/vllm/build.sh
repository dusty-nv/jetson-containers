#!/usr/bin/env bash
set -ex

uv pip install pre-commit nanobind==2.5.0
# Clone the repository if it doesn't exist
git clone --branch=${VLLM_BRANCH} --recursive --depth=1 https://github.com/vllm-project/vllm /opt/vllm ||
git clone --recursive --depth=1 https://github.com/vllm-project/vllm /opt/vllm

cd /opt/vllm
env

# cp /tmp/vllm/${VLLM_VERSION}.fa.diff /tmp/vllm/fa.diff
# git apply /tmp/vllm/${VLLM_VERSION}.diff

# echo "Applying vLLM CMake patches…"
#  [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "1" || "${IS_SBSA,,}" == "true" ]]; then
#   git apply -p1 /tmp/vllm/0.10.2.diff || echo "patch already applied"
# elif [[ "$(uname -m)" == "aarch64" && "${TORCH_CUDA_ARCH_LIST}" = "8.7" ]]; then
  # Generate and apply patch for Jetson SM 87,
  # required to apply flash attention path fa.diff.
  # Note, fa.diff was generated for vLLM commit 63b22e0dbb901b75619aa4bca2dfa1d7a71f439e :
  # - GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
  # - GIT_TAG a893712401d70362fbb299cd9c4b3476e8e9ed54
#  python3 /tmp/vllm/generate_diff.py                      # (re)generate the .diff files
#   if ! git apply -p1 /tmp/vllm/vllm_flash_attn.cmake.diff  ;then
#     echo "ERROR: Patch for SM 8.7 FAILED!" >&2
#     exit 1
#   else
#     echo "Patch for SM 8.7 applied successfully!"
#   fi
# fi
# File "/opt/venv/lib/python3.12/site-packages/gguf/gguf_reader.py"
# `newbyteorder` was removed from the ndarray class in NumPy 2.0
sed -i \
  -e 's|^gguf.*|gguf|g' \
  -e 's|^opencv-python-headless.*||g' \
  -e 's|^mistral_common.*|mistral_common|g' \
  -e 's|^compressed-tensors.*||g' \
  -e 's|^xgrammar.*||g' \
  requirements/common.txt

# Loosen flashinfer-python requirement to allow latest version.
sed -i \
  -e 's|^flashinfer-python.*|flashinfer-python|g' \
  requirements/cuda.txt

grep gguf requirements/common.txt


export USE_CUDNN=1
export VERBOSE=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"
export DG_JIT_USE_NVRTC=1 # DeepGEMM now supports NVRTC with up to 10x compilation speedup

python3 use_existing_torch.py || echo "skipping vllm/use_existing_torch.py"

uv pip install -r requirements/build.txt -v
python3 -m setuptools_scm

ARCH=$(uname -i)
if [ "${ARCH}" = "aarch64" ]; then
      export NVCC_THREADS=1
      export CUDA_NVCC_FLAGS="-Xcudafe --threads=1"
      export MAKEFLAGS='-j2'
      export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
      export NINJAFLAGS='-j2'
fi

uv build --wheel --no-build-isolation -v --out-dir /opt/vllm/wheels .
uv pip install /opt/vllm/wheels/vllm*.whl

cd /opt/vllm
uv pip install compressed-tensors

# Optionally upload to a repository using Twine
twine upload --verbose /opt/vllm/wheels/vllm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
