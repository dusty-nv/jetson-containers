#!/usr/bin/env bash
# auto_awq
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of autoawq ${AUTOAWQ_VERSION} with autoawq-kernels ${AUTOAWQ_KERNELS_VERSION}"
	exit 1
fi

uv pip install autoawq-kernels==${AUTOAWQ_KERNELS_VERSION} autoawq==${AUTOAWQ_VERSION}
