#!/usr/bin/env bash
set -ex

export MAX_JOBS=$(nproc) # this is for AGX (max 4 working on Orin NX)
export USE_CUDNN=1
export VERBOSE=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"

# Install dependencies of XGrammar
pip3 install pre-commit nanobind==2.5.0
git clone --depth=1 --recursive https://github.com/ai-dynamo/dynamo /opt/dynamo

echo "Building building vllm version ${AI-VLLM-DYNAMO_VERSION}..."
git clone --branch v"${VLLM_REF}"  --depth 1 https://github.com/vllm-project/vllm.git /opt/dynamo_vllm/
cd /opt/dynamo_vllm/
# Patch vLLM source with dynamo additions
patch -p1 < "/opt/dynamo/container/deps/vllm/${VLLM_PATCH}" || echo "Failed to apply patch ${VLLM_PATCH}"
sleep 5
sed -i 's/version("ai_dynamo_vllm")/version("vllm")/g' vllm/platforms/__init__.py 
sleep 5
if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" ]]; then
  echo "Applying vLLM CMake patches…"
  python3 /tmp/vllm/generate_diff.py                      # (re)generate the .diff files
  git apply -p1 /tmp/vllm/CMakeLists.txt.diff             # patch CMakeLists.txt
  git apply -p1 /tmp/vllm/vllm_flash_attn.cmake.diff      # patch vllm_flash_attn.cmake
else
  echo "SBSA build detected (IS_SBSA=${IS_SBSA}); skipping patch application."
fi
sleep 5
python3 use_existing_torch.py || echo "skipping vllm/use_existing_torch.py" 
pip3 install -r requirements/build.txt -v 
python3 -m setuptools_scm || echo "skipping vllm/setuptools_scm" 
pip3 wheel --no-build-isolation -v --wheel-dir=/opt/dynamo_vllm/wheels/ . 
pip3 install  /opt/dynamo_vllm/wheels/ai-dynamo-runtime*.whl

cd /opt/dynamo_vllm/
pip3 install compressed-tensors

# Optionally upload to a repository using Twine
twine upload --verbose /opt/dynamo_vllm/wheels/ai-dynamo-runtime*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
