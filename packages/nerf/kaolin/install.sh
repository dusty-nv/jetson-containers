#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of kaolin ${KAOLIN}"
	exit 1
fi

pip3 install kaolin==${KAOLIN_VERSION}