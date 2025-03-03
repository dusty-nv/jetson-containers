#!/usr/bin/env bash
# JAX builder for Jetson (architecture: ARM64, CUDA support)
set -ex

# Install LLVM/clang dev packages
./llvm.sh 18 all

echo "Building JAX for Jetson"

# Clone JAX repository
git clone --branch "jax-v${JAX_BUILD_VERSION}" --depth=1 --recursive https://github.com/google/jax /opt/jax || \
git clone --depth=1 --recursive https://github.com/google/jax /opt/jax

cd /opt/jax

# Build jaxlib from source with detected versions
BUILD_FLAGS='--disable_nccl '
BUILD_FLAGS+='--cuda_compute_capabilities="sm_87" '
BUILD_FLAGS+='--cuda_version=12.8.0 --cudnn_version=9.7.0 '
# BUILD_FLAGS+='--bazel_options=--repo_env=LOCAL_CUDA_PATH="/usr/local/cuda-12.8" '
# BUILD_FLAGS+='--bazel_options=--repo_env=LOCAL_CUDNN_PATH="/opt/nvidia/cudnn/" '
BUILD_FLAGS+='--output_path=/opt/wheels '

python3 build/build.py requirements_update
python3 build/build.py build $BUILD_FLAGS --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt

# Build the jax pip wheels
pip3 wheel --wheel-dir=/opt/wheels --no-deps --verbose .

# Upload the wheels to mirror
twine upload --verbose /opt/wheels/jaxlib-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/wheels/jax_cuda12_pjrt-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/wheels/jax_cuda12_plugin-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/wheels/jax-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container
cd /opt/wheels/
pip3 install jaxlib*.whl jax_cuda12_plugin*.whl jax_cuda12_pjrt*.whl opt_einsum
pip3 install --no-dependencies jax*.whl 
cd /opt/jax
