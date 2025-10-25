#!/usr/bin/env bash
set -ex

echo "Building AWQ ${AWQ_VERSION} (kernels=${AWQ_KERNEL_VERSION})"

git clone --branch=${AWQ_BRANCH} --depth=1 https://github.com/${AWQ_REPO} awq ||
git clone --depth=1 https://github.com/${AWQ_REPO} awq

sed -i \
  -e 's|torch==.*"|torch"|g' \
  -e 's|torchvision==.*"|torchvision"|g' \
  "awq/pyproject.toml"

sed -i 's/==/>=/g' "awq/pyproject.toml"

cat awq/pyproject.toml

uv build --wheel --no-build-isolation --out-dir $PIP_WHEEL_DIR --verbose ./awq
uv build --wheel --no-build-isolation --out-dir $PIP_WHEEL_DIR --verbose ./awq/awq/kernels

ls $PIP_WHEEL_DIR
rm -rf awq

uv pip install $PIP_WHEEL_DIR/awq*.whl
#uv pip show awq && python3 -c 'import awq' && python3 -m awq.entry --help

twine upload --verbose $PIP_WHEEL_DIR/awq-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose $PIP_WHEEL_DIR/awq_inference*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
