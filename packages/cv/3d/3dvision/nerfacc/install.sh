#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of nerfacc ${NERFACC_VERSION}"
	exit 1
fi

uv pip install nerfacc==${NERFACC_VERSION}
