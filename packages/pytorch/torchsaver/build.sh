#!/usr/bin/env bash
set -ex
echo "Building torchao ${TORCHAO_VERSION}"
   
git clone --branch v${TORCHAO_VERSION} --recursive --depth=1 https://github.com/fzyzcjy/torch_memory_saver /opt/torch_memory_saver || \
git clone --recursive --depth=1 https://github.com/fzyzcjy/torch_memory_saver /opt/torch_memory_saver

cd /opt/torch_memory_saver
#git checkout v${TORCHAO_VERSION}

#export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_TYPE=Release
sed -i -E 's/(version[[:space:]]*=[[:space:]]*version)[[:space:]]*\+[[:space:]]*version_suffix,/\1,/' setup.py
USE_CPP=1 python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/torch_memory_saver

pip3 install /opt/torch-memory-saver*.whl
pip3 show torch-memory-saver && python3 -c 'import torchao; print(torch-memory-saver.__version__);'

twine upload --verbose /opt/torch-memory-saver*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
