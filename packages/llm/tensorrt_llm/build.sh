#!/usr/bin/env bash
set -ex

echo "Building TensorRT-LLM ${TRT_LLM_VERSION}"

if [ -s ${SOURCE_TAR} ]; then
	echo "extracting TensorRT-LLM sources from ${TRT_LLM_SOURCE}"
	mkdir -p ${SOURCE_DIR}
	tar -xzf ${SOURCE_TAR} -C ${SOURCE_DIR}
else
	echo "cloning TensorRT-LLM sources from git (branch=${TRT_LLM_BRANCH})"
	git clone https://github.com/NVIDIA/TensorRT-LLM.git ${SOURCE_DIR}
	cd ${SOURCE_DIR}
	git checkout ${TRT_LLM_BRANCH}
	git status
	git submodule update --init --recursive
	git lfs pull
	
	if [ -s ${GIT_PATCHES} ]; then 
		echo "applying git patches from ${TRT_LLM_PATCH}"
		git apply ${GIT_PATCHES}
	fi
	
	sed -i 's|tensorrt.*||' requirements.txt
	sed -i 's|torch.*|torch|' requirements.txt
	sed -i 's|nvidia-cudnn.*||' requirements.txt
	
	git status
	git diff --submodule=diff
fi	

python3 ${SOURCE_DIR}/scripts/build_wheel.py \
        --clean \
        --build_type Release \
        --cuda_architectures "${CUDA_ARCHS}" \
        --build_dir ${BUILD_DIR} \
        --dist_dir /opt \
        --extra-cmake-vars "ENABLE_MULTI_DEVICE=OFF" \
        --benchmarks \
        --python_bindings

pip3 install --no-cache-dir --verbose /opt/tensorrt_llm*.whl

pip3 show tensorrt_llm
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

twine upload --verbose /opt/tensorrt_llm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
