#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of protomotions ${PROTOMOTIONS}"
	exit 1
fi

pip3 install protomotions==${PROTOMOTIONS_VERSION}