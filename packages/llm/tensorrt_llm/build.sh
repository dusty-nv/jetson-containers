#!/usr/bin/env bash
set -ex

echo "Building TensorRT-LLM ${TRT_LLM_VERSION} (branch=${TRT_LLM_BRANCH})"

# git-lfs is needed
apt-get update
apt-get install -y --no-install-recommends git-lfs
rm -rf /var/lib/apt/lists/*
apt-get clean

# clone the sources
git clone https://github.com/NVIDIA/TensorRT-LLM.git ${SRC_DIR}
cd ${SRC_DIR}

git checkout ${TRT_LLM_BRANCH}
git status

git submodule update --init --recursive
git lfs pull
  
# apply patches
if [ -s /tmp/tensorrt_llm/patch.diff ]; then 
	git apply /tmp/tensorrt_llm/patch.diff
fi

sed -i 's|tensorrt.*||' requirements.txt
sed -i 's|torch.*|torch|' requirements.txt

git status
git diff --submodule=diff
	
# build C++ and Python  
python3 ${SRC_DIR}/scripts/build_wheel.py \
        --clean \
        --build_type Release \
        --cuda_architectures "${CUDA_ARCHS}" \
        --build_dir ${BUILD_DIR} \
        --dist_dir /opt \
        --extra-cmake-vars "ENABLE_MULTI_DEVICE=OFF" \
        --benchmarks \
        --python_bindings

# install wheel
pip3 install --no-cache-dir --verbose /opt/tensorrt_llm*.whl

pip3 show tensorrt_llm
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

twine upload --verbose /opt/tvm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
