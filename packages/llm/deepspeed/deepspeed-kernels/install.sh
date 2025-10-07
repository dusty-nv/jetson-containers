#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of deepspeed-kernels ${DEEPSPEED_KERNELS_VERSION} (branch=${DEEPSPEED_KERNELS_BRANCH})"
	exit 1
fi

uv pip install deepspeed-kernels==${DEEPSPEED_KERNELS_VERSION}
