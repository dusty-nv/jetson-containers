#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of cutlass ${CUTLASS_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir nvidia-cutlass==${CUTLASS_VERSION} || \
pip3 install pycute || \
pip3 install nvidia-cutlass-dsl
