#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of fast-gaussian-rasterization ${FAST_GAUSSIAN}"
	exit 1
fi

pip3 install --no-cache-dir --verbose fast-gaussian-rasterization==${FAST_GAUSIAN_VERSION}