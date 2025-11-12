#!/usr/bin/env bash
# JAX builder for Jetson (architecture: ARM64, CUDA support)
set -ex

echo "Building JAX for Jetson"

# Clone JAX repository
git clone --branch "jax-v${JAX_BUILD_VERSION}" --depth=1 --recursive https://github.com/google/jax /opt/jax || \
git clone --depth=1 --recursive https://github.com/google/jax /opt/jax

cd /opt/jax

mkdir -p /opt/jax/wheels/
BUILD_FLAGS+='--clang_path=/usr/lib/llvm-21/bin/clang '
BUILD_FLAGS+='--output_path=/opt/jax/wheels/ '
# Build jaxlib from source with detected versions
if [ "${IS_SBSA}" -eq 1 ]; then
    echo "Building for SBSA architecture"
    BUILD_FLAGS+='--cuda_compute_capabilities="sm_87,sm_89,sm_90,sm_100,sm_110,sm_120,sm_121" '
    BUILD_FLAGS+='--cuda_version=13.0.0 --cudnn_version=9.12.0 '
    BUILD_FLAGS+='--bazel_options=--config=ci_linux_aarch64_cuda13 '
    BUILD_FLAGS+='--output_path=/opt/jax/wheels/ '
else
    echo "Building for non-SBSA architecture"
    BUILD_FLAGS+='--cuda_compute_capabilities="sm_87" '
    BUILD_FLAGS+='--cuda_version=12.6.0 --cudnn_version=9.3.0 '
fi

# Run the build
python3 build/build.py build $BUILD_FLAGS --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt

# Build the jax pip wheels
# uv build --wheel --out-dir /opt/jax/wheels/ --no-deps --verbose .

# Upload the wheels to mirror
twine upload --verbose /opt/jax/wheels/jaxlib-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax_cuda13_pjrt-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax_cuda13_plugin-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container
cd /opt/jax/wheels/
uv pip install jaxlib*.whl jax_cuda13_plugin*.whl jax_cuda13_pjrt*.whl opt_einsum
uv pip install --no-dependencies jax*.whl
cd /opt/jax
