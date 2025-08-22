#!/usr/bin/env bash
set -ex

# Install TVM from pip if available, else trigger build.sh

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of TVM from source (TVM_VERSION=${TVM_VERSION} TVM_COMMIT=${TVM_COMMIT})"
    exit 1
fi

if [ -n "${TVM_VERSION}" ]; then
    echo "Attempting pip install tvm==${TVM_VERSION}"
    pip3 install tvm==${TVM_VERSION} && exit 0 || echo "pip install tvm==${TVM_VERSION} failed, will build from source"
else
    echo "Attempting pip install latest tvm"
    pip3 install tvm && exit 0 || echo "pip install tvm failed, will build from source"
fi

exit 1




