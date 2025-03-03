#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pycolmap ${PYCOLMAP}"
	exit 1
fi

pip3 install pycolmap==${PYCOLMAP_VERSION}