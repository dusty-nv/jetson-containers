#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of DECORD ${decord}"
	exit 1
fi

pip3 install decord2==${DECORD_VERSION}
