#!/usr/bin/env bash
set -ex

echo "Building DeepSpeed ${DEEPSPEED_VERSION} (branch=${DEEPSPEED_BRANCH})"

git clone --branch=${DEEPSPEED_BRANCH} --depth=1 --recursive https://github.com/microsoft/DeepSpeed /opt/DeepSpeed ||
git clone --depth=1 --recursive https://github.com/microsoft/DeepSpeed /opt/DeepSpeed

cd /opt/DeepSpeed
# export DS_BUILD_CCL_COMM=0 --> skip the NCCL/CCL/RCCL “comm” op
export DS_BUILD_CCL_COMM=0 && \
python3 setup.py build_ext -j$(nproc) bdist_wheel --dist-dir $PIP_WHEEL_DIR

uv pip install $PIP_WHEEL_DIR/deepspeed-*.whl

twine upload --verbose $PIP_WHEEL_DIR/deepspeed-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
