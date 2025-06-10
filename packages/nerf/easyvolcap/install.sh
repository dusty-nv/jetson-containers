#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of fast_gauss ${EASYVOLCAP_VERSION}"
	exit 1
fi

pip3 install fast_gauss==${EASYVOLCAP_VERSION}