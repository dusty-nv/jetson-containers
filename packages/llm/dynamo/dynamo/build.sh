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

pip3 install maturin
pip3 install -U container/deps/requirements.txt
maturin develop --uvloop --release --cargo-extra-args="--release" --out /opt/dynamo/wheels
twine upload --verbose /opt/dynamo/wheels/ai-dynamo-runtime*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"

cd /opt/dynamo
pip3 wheel '.[all]' --wheel-dir=/opt/dynamo/wheels . --verbose
pip3 install /opt/dynamo/wheels/ai-dynamo*.whl

# Subida final
twine upload --verbose /opt/dynamo/wheels/ai-dynamo*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
