#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchcodec ${TORCHCODEC_VERSION}"
	exit 1
fi

pip3 install torchcodec~=${TORCHCODEC_VERSION} || \

