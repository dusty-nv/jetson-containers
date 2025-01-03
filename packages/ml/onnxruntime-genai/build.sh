#!/usr/bin/env bash
set -ex

# Extract ONNX Runtime version using Python
ONNXRUNTIME_VERSION=$(python3 -c "import onnxruntime as ort; print(ort.__version__)")

echo "Detected ONNX Runtime version: ${ONNXRUNTIME_VERSION}"
echo "Building onnxruntime-genai ${ONNXRUNTIME_GENAI_VERSION} (branch=${ONNXRUNTIME_GENAI_BRANCH})"

git clone --branch=rel-${ONNXRUNTIME_GENAI_VERSION} --depth=1 --recursive https://github.com/microsoft/onnxruntime-genai /opt/onnxruntime-genai || \
git clone --recursive https://github.com/microsoft/onnxruntime-genai /opt/onnxruntime-genai

cp /opt/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime*.so* ort/lib/

# Use the dynamically detected version for downloading ONNX Runtime headers
wget https://raw.githubusercontent.com/microsoft/onnxruntime/rel-${ONNXRUNTIME_VERSION}/include/onnxruntime/core/session/onnxruntime_c_api.h
wget https://raw.githubusercontent.com/microsoft/onnxruntime/rel-${ONNXRUNTIME_VERSION}/include/onnxruntime/core/session/onnxruntime_float16.h

# Use the dynamically detected version for symbolic linking
ln -s /home/jetson/Projects/onnxruntime-genai/ort/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /home/jetson/Projects/onnxruntime-genai/ort/lib/libonnxruntime.so

./build.sh --use_cuda --config Release --update --parallel --build --build_wheel \
        --skip_tests ${ONNXRUNTIME_FLAGS} \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable -I/usr/local/cuda/include" \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        --cmake_extra_defines CMAKE_INSTALL_PREFIX=${install_dir} \
        --cuda_home /usr/local/cuda --ort_home ./ort

twine upload --verbose /opt/onnxruntime-genai/build/Linux/Release/wheel/onnxruntime-genai*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
tarpack upload /opt/onnxruntime-genai/build/Linux/Release/wheel/onnxruntime-genai*.whl-${ONNXRUNTIME_GENAI_VERSION} || echo "failed to upload tarball"

# rm -rf /tmp/onnxruntime