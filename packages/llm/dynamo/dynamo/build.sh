#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${DYNAMO_VERSION} --depth=1 --recursive https://github.com/ai-dynamo/dynamo /opt/dynamo || \
git clone --depth=1 --recursive https://github.com/ai-dynamo/dynamo /opt/dynamo

cd /opt/dynamo
echo "Building ai-dynamo version ${DYNAMO_VERSION}..."
export CARGO_BUILD_JOBS=$(nproc)
export MAX_JOBS=$(nproc)

# Compilar con cargo
cargo build --release --features cuda,python

# Continuar con el resto del build
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

cd /opt/dynamo
pip3 wheel '.[all]' --wheel-dir=/opt/dynamo/wheels . --verbose
pip3 install /opt/dynamo/wheels/ai_dynamo*.whl

# Subida final
twine upload --verbose /opt/dynamo/wheels/ai_dynamo*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
