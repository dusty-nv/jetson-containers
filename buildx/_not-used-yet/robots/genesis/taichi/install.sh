#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of taichi ${TAICHI}"
	exit 1
fi

pip3 install taichi==${TAICHI_VERSION}