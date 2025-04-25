#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of minference ${MINFERENCE_VERSION}"
	exit 1
fi

pip3 install minference==${MINFERENCE_VERSION} || \
pip3 install minference==${MINFERENCE_VERSION_SPEC}