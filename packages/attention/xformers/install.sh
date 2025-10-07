#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of xformers ${XFORMERS}"
	exit 1
fi

uv pip install xformers==${XFORMERS_VERSION}
