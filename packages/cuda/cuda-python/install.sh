#!/usr/bin/env bash
set -ex

if [ "on" == "on" ]; then
	echo "Forcing build of cuda-python ${CUDA_PYTHON_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir cuda-python==${CUDA_PYTHON_VERSION}