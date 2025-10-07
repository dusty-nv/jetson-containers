#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of TVM ${TVM_VERSION}"
	exit 1
fi

uv pip install apache-tvm==${TVM_VERSION} ||
uv pip install --no-cache-dir tvm==${TVM_VERSION} ||
uv pip install pre apache-tvm==${TVM_VERSION} ||
uv pip install --no-cache-dir pre tvm==${TVM_VERSION} ||
echo "failed to install TVM ${TVM_VERSION}"
