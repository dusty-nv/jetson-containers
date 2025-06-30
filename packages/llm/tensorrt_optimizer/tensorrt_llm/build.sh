#!/usr/bin/env bash
set -ex

echo "Building TensorRT-LLM ${TRT_LLM_VERSION}"

git clone --recursive --depth 1 --branch ${TRT_LLM_VERSION} https://github.com/NVIDIA/TensorRT-LLM /opt/tensorrt_llm/ || \
git clone --recursive --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git /opt/tensorrt_llm/

cd /opt/tensorrt_llm/
git lfs pull
sed -i '/^diffusers[[:space:]=<>!]/d' requirements.txt
sed -i 's/==/>=/g' requirements.txt
pip3 install -r requirements.txt


python3 /opt/tensorrt_llm/scripts/build_wheel.py \
        --clean \
        --build_type Release \
        --cuda_architectures "${CUDA_ARCHS}" \
        --build_dir ${BUILD_DIR} \
        --dist_dir $PIP_WHEEL_DIR \
        --extra-cmake-vars "ENABLE_MULTI_DEVICE=0" \
        --benchmarks \
        --use_ccache \
        --python_bindings

pip3 install $PIP_WHEEL_DIR/tensorrt_llm*.whl

#pip3 show tensorrt_llm
#python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

twine upload --verbose $PIP_WHEEL_DIR/tensorrt_llm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
