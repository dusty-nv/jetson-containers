#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of easyvolcap ${EASYVOLCAP_VERSION}"
	exit 1
fi

uv pip install easyvolcap==${EASYVOLCAP_VERSION}
