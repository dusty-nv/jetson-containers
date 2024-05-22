#!/usr/bin/env bash
set -ex

echo " ================ Building openai_triron ${OPENAITRIRON_VERSION} ================"

pip3 uninstall -y triton

git clone --depth=1 https://github.com/openai/triton /opt/triton
git -C /opt/triton/third_party submodule update --init nvidia
sed -i \
    -e 's|LLVMAMDGPUCodeGen||g' \
    -e 's|LLVMAMDGPUAsmParser||g' \
    -e 's|-Werror|-Wno-error|g' \
    /opt/triton/CMakeLists.txt
pip3 wheel --wheel-dir=/opt --no-deps --verbose /opt/triton/python
rm -rf /opt/triton 

ls -l /opt/ 
pip3 install --no-cache-dir --verbose /opt/triton*.whl

pip3 show triton
python3 -c 'import triton'

twine upload --verbose /opt/triton*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
