#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of tinycudann ${TINYCUDANN}"
	exit 1
fi

uv pip install tinycudann==${TINYCUDANN_VERSION}
