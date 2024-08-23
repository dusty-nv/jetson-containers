#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of diffusers ${DIFFUSERS}"
	exit 1
fi

pip3 install --no-cache-dir --verbose diffusers==${DIFFUSERS_VERSION}
