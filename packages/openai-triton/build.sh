#!/usr/bin/env bash
set -ex

echo " ================ Building openai_triton ${OPENAITRITON_VERSION} ================"

pip3 uninstall -y triton

git clone --recursive --branch ${OPENAITRITON_BRANCH} --depth=1 https://github.com/openai/triton /opt/triton
cd /opt/triton
#git -C /opt/triton/third_party submodule update --init nvidia

sed -i \
    -e 's|LLVMAMDGPUCodeGen||g' \
    -e 's|LLVMAMDGPUAsmParser||g' \
    -e 's|-Werror|-Wno-error|g' \
    CMakeLists.txt
    
sed -i 's|^download_and_copy_ptxas|#|g' python/setup.py

mkdir -p third_party/cuda
ln -sf /usr/local/cuda/bin/ptxas $(pwd)/third_party/cuda/ptxas

pip3 wheel --wheel-dir=/opt --no-deps --verbose ./python

cd /
rm -rf /opt/triton 

pip3 install --no-cache-dir --verbose /opt/triton*.whl

pip3 show triton
python3 -c 'import triton'

twine upload --verbose /opt/triton*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
