#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of TVM ${TVM_VERSION}"
	exit 1
fi

pip3 install apache-tvm==${TVM_VERSION} ||
pip3 install --no-cache-dir tvm==${TVM_VERSION} ||
pip3 install pre apache-tvm==${TVM_VERSION} ||
pip3 install --no-cache-dir pre tvm==${TVM_VERSION} ||
echo "failed to install TVM ${TVM_VERSION}"
