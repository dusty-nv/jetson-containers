#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${DYNAMO_VERSION} --depth=1 --recursive https://github.com/ai-dynamo/dynamo /opt/dynamo || \
git clone --depth=1 --recursive https://github.com/ai-dynamo/dynamo /opt/dynamo


export MAX_JOBS=$(nproc)

echo "Building building vllm version ${VLLM_VERSION}..."
git clone --branch v"${VLLM_REF}"  --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm/vllm-"${VLLM_REF}" 
cd /tmp/vllm/vllm-"${VLLM_REF}/" 
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

echo "Building ai-dynamo version ${DYNAMO_VERSION}..."
export CARGO_BUILD_JOBS=$(nproc)
cd /opt/dynamo
cargo build --release --features cuda,python

echo "Building bindings for Python"
cd /opt/dynamo/lib/bindings/python
export NIXL_PREFIX=/usr/local/nixl
export NIXL_PLUGIN_DIR=/usr/local/nixl/lib/aarch64-linux-gnu/plugins
export LD_LIBRARY_PATH=$NIXL_PREFIX/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export RUSTFLAGS="-L $NIXL_PREFIX/lib/aarch64-linux-gnu"

pip3 install maturin
mkdir /opt/dynamo/wheels
pip3 install -U -r /opt/dynamo/container/deps/requirements.txt
maturin build --release --out /opt/dynamo/wheels
twine upload --verbose /opt/dynamo/wheels/ai_dynamo_runtime*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
pip3 install /opt/dynamo/wheels/ai_dynamo_runtime*.whl

cd /opt/dynamo
pip3 wheel '.[all]' --wheel-dir=/opt/dynamo/wheels . --verbose
pip3 install /opt/dynamo/wheels/ai_dynamo*.whl

twine upload --verbose /opt/dynamo/wheels/ai_dynamo*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
