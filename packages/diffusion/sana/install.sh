#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of sana ${SANA}"
	exit 1
fi

pip3 install --no-cache-dir --verbose sana==${SANA_VERSION}