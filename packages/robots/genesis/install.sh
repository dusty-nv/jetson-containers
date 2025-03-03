#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of genesis ${GENESIS}"
	exit 1
fi

pip3 install genesis-world==${GENESIS_VERSION}