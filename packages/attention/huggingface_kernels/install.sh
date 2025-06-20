#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of Hugging-Face Kernels ${KERNELS_VERSION}"
	exit 1
fi

pip3 install kernels==${KERNELS_VERSION}
pip3 show kernels && python3 -c 'import kernels'
