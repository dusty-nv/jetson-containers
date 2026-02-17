#!/usr/bin/env bash
set -ex
echo "Building TORCH_MEMORY_SAVER ${TORCH_MEMORY_SAVER_VERSION}"

git clone --branch v${TORCH_MEMORY_SAVER_VERSION} --recursive --depth=1 https://github.com/fzyzcjy/torch_memory_saver /opt/torch_memory_saver || \
git clone --recursive --depth=1 https://github.com/fzyzcjy/torch_memory_saver /opt/torch_memory_saver

cd /opt/torch_memory_saver
#git checkout v${TORCHAO_VERSION}

# Fix the package version
codebase_version=$(grep -Po "(?<=version=')[^']+" setup.py)
echo "Codebase version: ${codebase_version}"
echo "Wanted version: ${TORCH_MEMORY_SAVER_VERSION}"

if [ "$codebase_version" != "${TORCH_MEMORY_SAVER_VERSION}" ]; then
    sed -i "s/\(version='\)[^']*/\1${TORCH_MEMORY_SAVER_VERSION}/" setup.py
fi

#export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_TYPE=Release
export TMS_HOOK_MODE_TORCH=1

sed -i -E 's/(version[[:space:]]*=[[:space:]]*version)[[:space:]]*\+[[:space:]]*version_suffix,/\1,/' setup.py
USE_CPP=1 python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/torch_memory_saver

uv pip install /opt/torch_memory_saver*.whl
uv pip show torch_memory_saver && python3 -c 'import torch_memory_saver'

twine upload --verbose /opt/torch_memory_saver*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
