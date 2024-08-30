#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of nerfstudio ${NERFSTUDIO}"
	exit 1
fi

pip3 install --no-cache-dir --verbose --no-deps --no-build-isolation nerfstudio==${NERFSTUDIO_VERSION}