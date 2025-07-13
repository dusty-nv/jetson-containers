#!/usr/bin/env bash
# JAX builder for Jetson (architecture: ARM64, CUDA support)
set -ex

echo "Building JAX for Jetson"

# Clone JAX repository
git clone --branch "jax-v${JAX_BUILD_VERSION}" --depth=1 --recursive https://github.com/google/jax /opt/jax || \
git clone --depth=1 --recursive https://github.com/google/jax /opt/jax

cd /opt/jax

# Build jaxlib from source with detected versions
BUILD_FLAGS='--disable_nccl '
BUILD_FLAGS+='--cuda_compute_capabilities="sm_87,sm_89,sm_90,sm_100,sm_110,sm_121" '
BUILD_FLAGS+='--cuda_version=13.0.0 --cudnn_version=9.11.0 '
# BUILD_FLAGS+='--bazel_options=--repo_env=LOCAL_CUDA_PATH="/usr/local/cuda-12.8"'
# BUILD_FLAGS+='--bazel_options=--repo_env=LOCAL_CUDNN_PATH="/opt/nvidia/cudnn/"'
BUILD_FLAGS+='--output_path=$PIP_WHEEL_DIR '
BUILD_FLAGS+='--clang_path=/usr/lib/llvm-20/bin/clang'

python3 build/build.py requirements_update

# Start background process to monitor and patch matrix.h
(
    while true; do
        if [ -f "/root/.cache/bazel/_bazel_root/cfd1b2cc6fe180f3eb424db6004de364/external/cutlass_archive/include/cutlass/matrix.h" ]; then
            echo -e "\e[1;33m[PATCH] Found matrix.h, applying patch...\e[0m"
            sed -i 's/set_slice3x3/set_slice_3x3/g' /root/.cache/bazel/_bazel_root/cfd1b2cc6fe180f3eb424db6004de364/external/cutlass_archive/include/cutlass/matrix.h
            echo -e "\e[1;32m[PATCH] Patch applied successfully!\e[0m"
            break
        fi
        sleep 1
    done
) &

# Run the build
python3 build/build.py build $BUILD_FLAGS --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt

# Build the jax pip wheels
pip3 wheel --wheel-dir=$PIP_WHEEL_DIR --no-deps --verbose .

# Upload the wheels to mirror
twine upload --verbose $PIP_WHEEL_DIR/jaxlib-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose $PIP_WHEEL_DIR/jax_cuda12_pjrt-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose $PIP_WHEEL_DIR/jax_cuda12_plugin-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose $PIP_WHEEL_DIR/jax-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container
cd $PIP_WHEEL_DIR/
pip3 install jaxlib*.whl jax_cuda12_plugin*.whl jax_cuda12_pjrt*.whl opt_einsum
pip3 install --no-dependencies jax*.whl
cd /opt/jax
