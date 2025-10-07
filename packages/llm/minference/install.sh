#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of minference ${MINFERENCE_VERSION}"
	exit 1
fi
uv pip install tilelang
uv pip install minference==${MINFERENCE_VERSION} || \
uv pip install minference==${MINFERENCE_VERSION_SPEC}
