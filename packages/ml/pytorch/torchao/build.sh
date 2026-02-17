#!/usr/bin/env bash
set -ex
echo "Building torchao ${TORCHAO_VERSION}"

git clone --branch v${TORCHAO_VERSION} --recursive --depth=1 https://github.com/pytorch/ao /opt/torchao || \
git clone --recursive --depth=1 https://github.com/pytorch/ao /opt/torchao

cd /opt/torchao
#git checkout v${TORCHAO_VERSION}

#export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_TYPE=Release
sed -i -E 's/(version[[:space:]]*=[[:space:]]*version)[[:space:]]*\+[[:space:]]*version_suffix,/\1,/' setup.py
USE_CPP=1 python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/torchao

uv pip install /opt/torchao*.whl
uv pip show torchao && python3 -c 'import torchao; print(torchao.__version__);'

twine upload --verbose /opt/torchao*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
