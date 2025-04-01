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
