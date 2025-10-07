#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of ktransformers ${KTRANSFORMERS_VERSION}"
	exit 1
fi

uv pip install compressed-tensors ktransformers==${KTRANSFORMERS_VERSION}
