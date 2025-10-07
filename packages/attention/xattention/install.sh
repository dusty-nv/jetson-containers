#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of xattention ${XATTENTION_VERSION}"
	exit 1
fi

uv pip install xattn==${XATTENTION_VERSION}
