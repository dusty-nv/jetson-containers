#!/usr/bin/env bash
set -ex

echo "Building deepspeed-kernels ${DEEPSPEED_KERNELS_VERSION} (branch=${DEEPSPEED_KERNELS_BRANCH})"

git -C /opt clone --branch=${DEEPSPEED_KERNELS_BRANCH} --depth=1 --recursive https://github.com/microsoft/DeepSpeed-Kernels
cd /opt/DeepSpeed-Kernels

CUDA_ARCH_LIST=${CUDA_ARCHITECTURES} python3 setup.py --verbose build_ext -j$(nproc) bdist_wheel --dist-dir /opt/wheels

pip3 install --no-cache-dir --verbose /opt/wheels/deepspeed_kernels*.whl

twine upload --verbose /opt/wheels/deepspeed_kernels*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
