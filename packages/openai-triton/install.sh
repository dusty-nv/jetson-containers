#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of openai_triton ${OPENAITRITON_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir --verbose triton==${OPENAITRITON_VERSION}
