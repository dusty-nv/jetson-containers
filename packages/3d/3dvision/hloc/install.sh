#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of hloc ${HLOC}"
	exit 1
fi

uv pip install hloc==${HLOC_VERSION}
