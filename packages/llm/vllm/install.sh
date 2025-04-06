#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of vllm ${VLLM_VERSION}"
	exit 1
fi

pip3 install \
	compressed-tensors \
	xgrammar \
	vllm==${VLLM_VERSION}

# File "/opt/venv/lib/python3.12/site-packages/gguf/gguf_reader.py"
# `newbyteorder` was removed from the ndarray class in NumPy 2.0
if [ $NUMPY_VERSION_MAJOR -ge 2 ]; then
	pip3 install 'gguf>=0.13.0'
fi