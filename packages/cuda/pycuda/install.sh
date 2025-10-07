#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pycuda ${PYCUDA_VERSION}"
	exit 1
fi

uv pip install pycuda==${PYCUDA_VERSION}
uv pip show pycuda && python3 -c 'import pycuda; print(pycuda.VERSION_TEXT)'
