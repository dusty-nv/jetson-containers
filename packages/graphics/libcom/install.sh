#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of libcom ${LIBCOM}"
	exit 1
fi

pip3 install --no-cache-dir --verbose libcom==${LIBCOM_VERSION}