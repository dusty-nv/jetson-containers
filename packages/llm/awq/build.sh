#!/usr/bin/env bash
set -ex

echo "Building AWQ ${AWQ_VERSION} (kernels=${AWQ_KERNEL_VERSION})"

git clone --branch=${AWQ_BRANCH} --depth=1 https://github.com/mit-han-lab/llm-awq awq

pip3 wheel --wheel-dir=/opt/wheels --verbose ./awq
pip3 wheel --wheel-dir=/opt/wheels --verbose ./awq/awq/kernels

ls /opt/wheels
rm -rf awq

pip3 install --no-cache-dir --verbose /opt/wheels/awq*.whl
pip3 show awq && python3 -c 'import awq' && python3 -m awq.entry --help

twine upload --verbose /opt/wheels/awq-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/wheels/awq_inference*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"