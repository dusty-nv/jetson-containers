#!/usr/bin/env bash
set -ex

echo "Building TensorRT-LLM ${TRT_LLM_VERSION}"

python3 ${SOURCE_DIR}/scripts/build_wheel.py \
        --clean \
        --build_type Release \
        --cuda_architectures "${CUDA_ARCHS}" \
        --build_dir ${BUILD_DIR} \
        --dist_dir /opt \
        --extra-cmake-vars "ENABLE_MULTI_DEVICE=0" \
        --benchmarks \
        --python_bindings

pip3 install --no-cache-dir --verbose /opt/tensorrt_llm*.whl

#pip3 show tensorrt_llm
#python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

twine upload --verbose /opt/tensorrt_llm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
