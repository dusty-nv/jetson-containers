#!/usr/bin/env bash
set -ex

# Extract ONNX Runtime version using Python
ONNXRUNTIME_VERSION=$(python3 -c "import onnxruntime as ort; print(ort.__version__)")

echo "CUDA Version: ${CUDA_VERSION}"
echo "Detected ONNX Runtime version: ${ONNXRUNTIME_VERSION}"
echo "Building onnxruntime_genai ${ONNXRUNTIME_GENAI_VERSION} (branch=${ONNXRUNTIME_GENAI_BRANCH})"

git clone --branch=rel-${ONNXRUNTIME_GENAI_VERSION} --depth=1 --recursive https://github.com/microsoft/onnxruntime-genai /opt/onnxruntime_genai || \
git clone --recursive https://github.com/microsoft/onnxruntime-genai /opt/onnxruntime_genai

mkdir -p /opt/onnxruntime_genai/ort/lib/
mkdir -p /opt/onnxruntime_genai/ort/include/

cp /opt/venv/lib/python${PYTHON_VERSION}/site-packages/onnxruntime/capi/libonnxruntime*.so* /opt/onnxruntime_genai/ort/lib/
cd /opt/onnxruntime_genai/ort/include/
# Use the dynamically detected version for downloading ONNX Runtime headers
wget https://raw.githubusercontent.com/microsoft/onnxruntime/rel-${ONNXRUNTIME_VERSION}/include/onnxruntime/core/session/onnxruntime_c_api.h
wget https://raw.githubusercontent.com/microsoft/onnxruntime/rel-${ONNXRUNTIME_VERSION}/include/onnxruntime/core/session/onnxruntime_float16.h

# Use the dynamically detected version for symbolic linking
ln -s /opt/onnxruntime_genai/ort/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /opt/onnxruntime_genai/ort/lib/libonnxruntime.so

cd /opt/onnxruntime_genai

install_dir="/opt/onnxruntime_genai/install"

./build.sh --use_cuda --config Release --update --parallel --build \
        --skip_tests ${ONNXRUNTIME_FLAGS} \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable -I/usr/local/cuda/include" \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        --cmake_extra_defines CMAKE_INSTALL_PREFIX=${install_dir} \
        --cuda_home /usr/local/cuda --ort_home ./ort

cd build/Linux/Release
make install

find / -type f -name "onnxruntime_genai*.whl" -exec cp {} /opt/ \; 2>/dev/null

uv pip install /opt/onnxruntime_genai*.whl
python3 -c 'import onnxruntime_genai; print(onnxruntime_genai.__version__);'

twine upload --verbose /opt/onnxruntime_genai*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
tarpack upload onnxruntime_genai-${ONNXRUNTIME_GENAI_VERSION} ${install_dir} || echo "failed to upload tarball"
