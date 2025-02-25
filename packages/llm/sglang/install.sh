#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of sglang ${SGLANG_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir --verbose sglang==${SGLANG_VERSION}
