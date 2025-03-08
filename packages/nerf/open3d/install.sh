#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of nerfacc ${NERFACC}"
	exit 1
fi

pip3 install nerfacc==${NERFACC_VERSION}