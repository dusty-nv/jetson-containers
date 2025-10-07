#!/usr/bin/env bash
set -ex

echo "Building Apache TVM from source (commit=${TVM_FFI_COMMIT})"

export TVM_FFI_SRC_DIR=/opt/tvm_ffi

git clone --recursive --depth=1 --branch=TVM_FFI_COMMIT https://github.com/apache/tvm ${TVM_FFI_SRC_DIR} ||
git clone --recursive --depth=1 https://github.com/apache/tvm
cd ${TVM_FFI_SRC_DIR}

cd /opt/tvm/python
uv build --wheel --out-dir /opt/tvm_ffi/wheels .

uv pip install /opt/tvm_ffi/wheels/apache_tvm_ffi-*.whl

twine upload --verbose /opt/tvm_ffi/wheels/apache_tvm_ffi-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
