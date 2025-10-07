#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of gsplat ${GSPLAT}"
	exit 1
fi

uv pip install gsplat==${GSPLAT_VERSION}
