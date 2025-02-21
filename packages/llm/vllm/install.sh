#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of vllm ${VLLM_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir --verbose compressed-tensors triton==3.1.0 xgrammar bitsandbytes xformers flash-attn vllm==${VLLM_VERSION}
