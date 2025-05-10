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
cargo build --features cuda --release

# GUARDAR HASTA AQUÍ: crear una imagen con el estado actual
# Esto asume que este script está corriendo en un contenedor

# Guardar el estado actual del contenedor en una imagen temporal
CONTAINER_ID=$(hostname)
docker commit "$CONTAINER_ID" "ai-dynamo:post-cargo-build"

echo "Imagen intermedia guardada como ai-dynamo:post-cargo-build"

# Continuar con el resto del build
echo "Building bindings for Python"
cd lib/bindings/python
pip3 wheel --wheel-dir=/opt/dynamo/wheels . --verbose
pip3 install /opt/dynamo/wheels/ai-dynamo-runtime*.whl
twine upload --verbose /opt/dynamo/wheels/ai-dynamo-runtime*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"

cd /opt/dynamo
pip3 wheel '.[all]' --wheel-dir=/opt/dynamo/wheels . --verbose
pip3 install /opt/dynamo/wheels/ai-dynamo*.whl

# Subida final
twine upload --verbose /opt/dynamo/wheels/ai-dynamo*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
