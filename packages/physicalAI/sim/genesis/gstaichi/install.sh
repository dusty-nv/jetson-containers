#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of gstaichi ${GSTAICHI_VERSION}"
	exit 1
fi

uv pip install gstaichi==${GSTAICHI_VERSION}
