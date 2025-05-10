#!/usr/bin/env bash
set -ex

# Install dependencies of XGrammar
pip3 install pre-commit nanobind==2.5.0
git clone --depth=1 --recursive https://github.com/ai-dynamo/dynamo /opt/dynamo

echo "Building building vllm version ${AI-VLLM-DYNAMO_VERSION}..."
git clone --branch v"${VLLM_REF}"  --depth 1 https://github.com/vllm-project/vllm.git /opt/ai-dynamo-vllm/vllm-"${VLLM_REF}"
cd /opt/ai-dynamo-vllm/vllm/vllm-"${VLLM_REF}/" 
# Patch vLLM source with dynamo additions
patch -p1 < "/opt/dynamo/container/deps/vllm/${VLLM_PATCH}" || echo "Failed to apply patch ${VLLM_PATCH}"
sleep 5
sed -i 's/version("ai_dynamo_vllm")/version("vllm")/g' vllm/platforms/__init__.py 
sleep 5
python3 use_existing_torch.py || echo "skipping vllm/use_existing_torch.py" 
pip3 install -r requirements/build.txt -v 
python3 -m setuptools_scm || echo "skipping vllm/setuptools_scm" 
pip3 wheel --no-build-isolation -v --wheel-dir=/opt/ . 
pip3 install  /opt/ai-dynamo-runtime*.whl \
twine upload --verbose /opt/ai-dynamo-runtime*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"



# File "/opt/venv/lib/python3.12/site-packages/gguf/gguf_reader.py"
# `newbyteorder` was removed from the ndarray class in NumPy 2.0
sed -i 's|gguf.*|gguf|g' requirements/common.txt
grep gguf requirements/common.txt

export MAX_JOBS=$(nproc) # this is for AGX (max 4 working on Orin NX)
export USE_CUDNN=1
export VERBOSE=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"

python3 use_existing_torch.py || echo "skipping vllm/use_existing_torch.py"

pip3 install -r requirements/build.txt -v
python3 -m setuptools_scm
pip3 wheel --no-build-isolation -v --wheel-dir=/opt/vllm/wheels .
pip3 install /opt/vllm/wheels/vllm*.whl

cd /opt/vllm
pip3 install compressed-tensors

# Optionally upload to a repository using Twine
twine upload --verbose /opt/vllm/wheels/vllm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/xgrammar/wheels/xgrammar*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
