#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of fast_gauss ${FAST_GAUSS}"
	exit 1
fi

uv pip install fast_gauss==${FAST_GAUSS_VERSION}
