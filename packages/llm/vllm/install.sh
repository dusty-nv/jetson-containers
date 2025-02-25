#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of vllm ${VLLM_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir --verbose compressed-tensors vllm==${VLLM_VERSION}
