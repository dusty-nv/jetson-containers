#!/usr/bin/env bash
set -ex

echo "Building FlashAttention ${FLASH_ATTENTION_VERSION}"

git clone --depth=1 --branch=v${FLASH_ATTENTION_VERSION} https://github.com/Dao-AILab/flash-attention /opt/flash-attention

cd /opt/flash-attention

git apply /tmp/flash-attention/patch.diff
git diff
git status
 
FLASH_ATTENTION_FORCE_BUILD=1 \
FLASH_ATTENTION_FORCE_CXX11_ABI=0 \
FLASH_ATTENTION_SKIP_CUDA_BUILD=0 \
MAX_JOBS=$(nproc) \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

ls /opt
cd /

pip3 install --no-cache-dir --verbose /opt/flash_attn*.whl
#pip3 show flash-attn && python3 -c 'import flash_attn'

twine upload --verbose /opt/flash_attn*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
