#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of gsplat ${GSPLAT}"
	exit 1
fi

pip3 install gsplat==${GSPLAT_VERSION}