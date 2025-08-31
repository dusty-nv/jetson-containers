#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of TVM ${TVM_VERSION}"
	exit 1
fi

pip3 install tvm==${TVM_VERSION}
