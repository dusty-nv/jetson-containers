#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of quadrants ${QUADRANTS_VERSION}"
	exit 1
fi

uv pip install quadrants==${QUADRANTS_VERSION}
