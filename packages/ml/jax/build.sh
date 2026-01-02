#!/usr/bin/env bash
# JAX builder for Jetson (architecture: ARM64, CUDA support)
set -ex

echo "Building JAX for Jetson"

# Clone JAX repository
# Note: JAX versions are typically v0.4.x. If v0.9.0 doesn't exist, this falls back to main.
git clone --branch "jax-v${JAX_BUILD_VERSION}" --depth=1 --recursive https://github.com/google/jax /opt/jax || \
git clone --depth=1 --recursive https://github.com/google/jax /opt/jax

cd /opt/jax

mkdir -p /opt/jax/wheels/

# Initialize flags
BUILD_FLAGS="--clang_path=/usr/lib/llvm-21/bin/clang --output_path=/opt/jax/wheels/ "

if [ "${IS_SBSA}" -eq 1 ]; then
    echo "Building for SBSA architecture"
    BUILD_FLAGS+='--cuda_compute_capabilities="sm_87,sm_89,sm_90,sm_100,sm_110,sm_120,sm_121" '
    BUILD_FLAGS+='--cuda_version=13.0.2 --cudnn_version=9.16.0 '

    # --- BAZEL CONFIGURATION ---
    # 2. Fix Abseil: Disable nullability attributes.
    #    Clang 20+ enables them, but Abseil's headers place them incorrectly, causing syntax errors.
    BUILD_FLAGS+='--bazel_options=--copt=-DABSL_HAVE_NULLABILITY_ATTRIBUTES=0 '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-DABSL_HAVE_NULLABILITY_ATTRIBUTES=0 '
    BUILD_FLAGS+='--bazel_options=--copt=-D_Nullable= '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-D_Nullable= '
    BUILD_FLAGS+='--bazel_options=--copt=-D_Nonnull= '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-D_Nonnull= '

    # 3. Fix Protobuf: Polyfill the missing __is_bitwise_cloneable builtin.
    #    We alias it to __is_trivially_copyable which is supported.
    BUILD_FLAGS+='--bazel_options=--copt=-D__is_bitwise_cloneable=__is_trivially_copyable '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-D__is_bitwise_cloneable=__is_trivially_copyable '

    # 4. Fix Abseil: Polyfill the missing __builtin_is_cpp_trivially_relocatable builtin.
    #    NVCC/Clang interaction causes this builtin to be undefined despite passing feature checks.
    #    We alias it to __is_trivially_copyable, which is a safe fallback for build purposes.
    BUILD_FLAGS+='--bazel_options=--copt=-D__builtin_is_cpp_trivially_relocatable=__is_trivially_copyable '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-D__builtin_is_cpp_trivially_relocatable=__is_trivially_copyable '

else
    echo "Building for non-SBSA architecture"
    BUILD_FLAGS+='--cuda_compute_capabilities="sm_87" '
    BUILD_FLAGS+='--cuda_version=12.6.0 --cudnn_version=9.3.0 '
fi

# Run the build
# Note: $BUILD_FLAGS is unquoted to allow word splitting of the individual bazel arguments
python3 build/build.py build $BUILD_FLAGS --wheels=jax,jaxlib,jax-cuda-plugin,jax-cuda-pjrt

# Upload the wheels to mirror
twine upload --verbose /opt/jax/wheels/jaxlib-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jaxlib-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax_cuda13_pjrt-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax_cuda13_plugin-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container
cd /opt/jax/wheels/
uv pip install jaxlib*.whl jax_cuda13_plugin*.whl jax_cuda13_pjrt*.whl opt_einsum
uv pip install jax
cd /opt/jax
