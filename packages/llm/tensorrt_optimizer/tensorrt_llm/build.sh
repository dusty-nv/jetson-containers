#!/usr/bin/env bash
set -ex

echo "Building TensorRT-LLM ${TRT_LLM_VERSION}"

# Apply git patches if present (GIT_PATCHES is set from TRT_LLM_PATCH in Dockerfile)
if [ -s "${GIT_PATCHES}" ]; then
    echo "Applying git patches from ${GIT_PATCHES}"
    git apply "${GIT_PATCHES}" || echo "Warning: Patch may have already been applied"
fi

REQUIREMENTS_FILENAME="requirements.txt"
DEV_REQUIREMENTS_FILENAME="requirements-dev.txt"

if [[ "${TRT_LLM_BRANCH}" == *"jetson"* ]]; then
    REQUIREMENTS_FILENAME="requirements-jetson.txt"
    DEV_REQUIREMENTS_FILENAME="requirements-dev-jetson.txt"
fi

     
sed -i '/^diffusers[[:space:]=<>!]/d' "${REQUIREMENTS_FILENAME}"
sed -i 's/==/>=/g' "${REQUIREMENTS_FILENAME}"
sed -i 's/cuda-python.*/cuda-python/g' "${REQUIREMENTS_FILENAME}"
sed -i 's|flashinfer-python.*|flashinfer-python|' "${REQUIREMENTS_FILENAME}"
sed -i 's|^torch.*|torch|' "${REQUIREMENTS_FILENAME}"
sed -i 's|typing-extensions.*|typing-extensions|' "${DEV_REQUIREMENTS_FILENAME}"

uv pip install -r "${REQUIREMENTS_FILENAME}" 
uv pip install -r "${DEV_REQUIREMENTS_FILENAME}"

# Install TensorRT Wheel First to ensure libs are present
TRT_WHEEL=$(find /usr -name "tensorrt-*-cp310-*-linux_aarch64.whl" -print -quit)

if [ -f "$TRT_WHEEL" ]; then
    echo "Installing existing TensorRT wheel: $TRT_WHEEL"
    uv pip install "$TRT_WHEEL"
else
    echo "CRITICAL: TensorRT wheel not found. Build cannot proceed."
    exit 1
fi

echo "Configuring build environment to use existing TensorRT..."

# Create compatibility layout for TensorRT using PIP installed libs
if [ ! -d "/usr/local/tensorrt" ]; then
    echo "Creating compatibility layout at /usr/local/tensorrt..."
    mkdir -p /usr/local/tensorrt/include
    mkdir -p /usr/local/tensorrt/lib

    # Find where libnvinfer is installed on the system (e.g. /usr/lib/aarch64-linux-gnu)
    # We prioritize the system root /usr
    LIBNVINFER_PATH=$(find /usr -name "libnvinfer.so.*" 2>/dev/null | head -n 1)
    
    if [ -n "$LIBNVINFER_PATH" ]; then
        TRT_SYS_LIB_DIR=$(dirname "$LIBNVINFER_PATH")
        echo "Found system TensorRT libraries at: $TRT_SYS_LIB_DIR"
        
        # Symlink libraries to the compatibility directory
        # We assume if we found one, others are there too.
        ln -sf "$TRT_SYS_LIB_DIR"/libnvinfer* /usr/local/tensorrt/lib/
        ln -sf "$TRT_SYS_LIB_DIR"/libnvonnx*  /usr/local/tensorrt/lib/
        ln -sf "$TRT_SYS_LIB_DIR"/libnvpar*   /usr/local/tensorrt/lib/ 2>/dev/null || true
        # Also symlink plugin library if present (it might be in a different package or same dir)
        ln -sf "$TRT_SYS_LIB_DIR"/libnvinfer_plugin* /usr/local/tensorrt/lib/ 2>/dev/null || true
    else
        echo "WARNING: Could not find libnvinfer.so.* in /usr. Build commonly fails if libraries are missing."
    fi


    # But previous error was LIBRARY missing, implying headers MIGHT be found?
    # CMake said "found suitable version 10.3.0.26". This implies it found NvInferVersion.h
    # Where? Likely /usr/include/aarch64-linux-gnu (from base image).
    # So we keep the header symlinks from system just in case.
    if [ -d "/usr/include/aarch64-linux-gnu" ]; then
        ln -sf /usr/include/aarch64-linux-gnu/NvInfer* /usr/local/tensorrt/include/ || true
        ln -sf /usr/include/aarch64-linux-gnu/NvOnnx* /usr/local/tensorrt/include/ || true
    fi
    if [ -f "/usr/include/NvInfer.h" ]; then
         ln -sf /usr/include/Nv* /usr/local/tensorrt/include/
    fi

    # 3. Create Development Symlinks (libnvinfer.so -> libnvinfer.so.10)
    cd /usr/local/tensorrt/lib
    
    for LIB in libnvinfer libnvonnxparser libnvinfer_plugin; do
        if [ ! -f "${LIB}.so" ]; then
             TARGET=$(ls ${LIB}.so.* | head -n 1)
             if [ -n "$TARGET" ]; then
                 ln -sf "$TARGET" "${LIB}.so"
                 echo "Created dev link: ${LIB}.so -> $TARGET"
             else
                 echo "Warning: Could not find versioned lib for ${LIB}"
             fi
        fi
    done
    
    cd ${SOURCE_DIR}

    # Compatibility for lib64 search
    ln -sf lib /usr/local/tensorrt/lib64
    
    echo "Listing TRT libs in /usr/local/tensorrt/lib (FINAL):"
    ls -l /usr/local/tensorrt/lib
fi

export LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="/usr/local/tensorrt:${CMAKE_PREFIX_PATH}"


CUTLASS_PYTHON_DIR="3rdparty/cutlass/python"

if [ -d "$CUTLASS_PYTHON_DIR" ]; then
    cd "$CUTLASS_PYTHON_DIR"
    
    if [ ! -f "setup.py" ]; then
        ln -sf setup_library.py setup.py
    fi

    echo "Installing CUTLASS into venv..."
    uv pip install . || exit 1

    echo "Injecting skip logic into setup script..."
    
    sed -i '/def perform_setup():/a \    print("CUTLASS already installed via build.sh. Skipping internal setup."); return' setup_library.py
    
    grep "Skipping internal setup" setup_library.py || echo "WARNING: Sed injection might have failed"

    cd ${SOURCE_DIR}
else
    echo "CRITICAL: CUTLASS directory missing!"
    exit 1
fi

# Patched: removed --python_bindings as it is not supported in this version
# Patched: added --trt_root to point to compatibility layout
python3 ${SOURCE_DIR}/scripts/build_wheel.py \
        --clean \
        --build_type Release \
        --cuda_architectures "${CUDA_ARCHS}" \
        --build_dir ${BUILD_DIR} \
        --dist_dir $PIP_WHEEL_DIR \
        --extra-cmake-vars "ENABLE_MULTI_DEVICE=0;TORCH_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1" \
        --benchmarks \
        --use_ccache \
        --trt_root /usr/local/tensorrt

uv pip install $PIP_WHEEL_DIR/tensorrt_llm*.whl

#uv pip show tensorrt_llm
#python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

twine upload --verbose $PIP_WHEEL_DIR/tensorrt_llm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
