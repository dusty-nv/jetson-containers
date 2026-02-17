#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pixsfm ${PIXSFM}"
	exit 1
fi

uv pip install pixsfm==${PIXSFM_VERSION}
