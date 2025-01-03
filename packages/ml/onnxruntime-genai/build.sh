#!/usr/bin/env bash
set -ex

echo "Building onnxruntime-genai ${ONNXRUNTIME_GENAI_VERSION} (branch=${ONNXRUNTIME_GENAI_BRANCH})"

git clone --recursive --depth=1 https://github.com/microsoft/onnxruntime-genai /opt/onnxruntime-genai
cd /opt/onnxruntime

git clone --branch=rel-${ONNXRUNTIME_GENAI_VERSION} --depth=1 --recursive https://github.com/microsoft/onnxruntime-genai /opt/onnxruntime-genai || \
git clone --recursive https://github.com/microsoft/onnxruntime-genai /opt/onnxruntime-genai

./build.sh --config Release --update --parallel --build --build_wheel --build_shared_lib \
        --skip_tests --skip_submodule_sync ${ONNXRUNTIME_FLAGS} \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable -I/usr/local/cuda/include" \
	   --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
	   --cmake_extra_defines CMAKE_INSTALL_PREFIX=${install_dir} \
	   --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/$(uname -m)-linux-gnu \
        --use_tensorrt --tensorrt_home /usr/lib/$(uname -m)-linux-gnu


twine upload --verbose /opt/onnxruntime-genai*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
tarpack upload onnxruntime-gpu-${ONNXRUNTIME_GENAI_VERSION} ${install_dir} || echo "failed to upload tarball"

#rm -rf /tmp/onnxruntime
