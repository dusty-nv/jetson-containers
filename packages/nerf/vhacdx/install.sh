#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of vhacdx ${VHACDX}"
	exit 1
fi

pip3 install vhacdx==${VHACDX_VERSION}