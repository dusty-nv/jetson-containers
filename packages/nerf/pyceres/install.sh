#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pyceres ${PYCERES}"
	exit 1
fi

pip3 install pyceres==${PYCERES_VERSION}