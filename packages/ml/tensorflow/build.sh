#!/usr/bin/env bash
# TensorFlow builder for Jetson (architecture: ARM64, CUDA support)
set -ex

echo "Building Tensorflow ${TENSORFLOW_VERSION}"

echo "Building TensorFlow for Jetson"

# Clone the TensorFlow repository
git clone --branch "v${TENSORFLOW_VERSION}" --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow || \
git clone --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow

cd /opt/tensorflow
mkdir -p /opt/tensorflow/wheels/

# Initialize flags
BUILD_FLAGS="--clang_path=/usr/lib/llvm-21/bin/clang --output_path=/opt/tensorflow/wheels/ "

if [ "${IS_SBSA}" -eq 1 ]; then
    echo "Building for SBSA architecture"
    BUILD_FLAGS+='--cuda_compute_capabilities="sm_87,sm_89,sm_90,sm_100,sm_110,sm_120,sm_121" '
    BUILD_FLAGS+='--cuda_version=13.1.0 --cudnn_version=9.17.0 '

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


# Build the TensorFlow pip package
bazel build //tensorflow/tools/pip_package:wheel --action_env CLANG_CUDA_COMPILER_PATH="/usr/lib/llvm-21/bin/clang" --config=cuda_clang --repo_env=WHEEL_NAME=tensorflow --config=cuda --config=cuda_wheel --config=nonccl --copt=-Wno-sign-compare --copt=-Wno-gnu-offsetof-extensions --copt=-Wno-error=unused-command-line-argument

# Upload the wheels to mirror
twine upload --verbose /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container
uv pip install /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl

mv /opt/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl /opt/tensorflow/
# Clean up all files and close bazel server
bazel clean --expunge

cd /opt/
