#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of glomap ${GLOMAP}"
	exit 1
fi

pip3 install glomap==${GLOMAP_VERSION}