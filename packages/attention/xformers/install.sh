#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of xformers ${XFORMERS}"
	exit 1
fi

pip3 install xformers==${XFORMERS_VERSION}
