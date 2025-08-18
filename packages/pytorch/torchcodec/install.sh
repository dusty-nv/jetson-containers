#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchcodec ${TORCHCODEC_VERSION}"
	exit 1
fi

pip3 install torchcodec~=${TORCHCODEC_VERSION} || \
pip3 install torchcodec~=${TORCHCODEC_VERSION} || { echo "Failed to install torchcodec version ${TORCHCODEC_VERSION}"; exit 1; }
