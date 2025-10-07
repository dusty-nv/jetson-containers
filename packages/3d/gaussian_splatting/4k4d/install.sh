#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of easyvolcap ${FOURkFOUR_VERSION}"
	exit 1
fi

uv pip install easyvolcap==${FOURkFOUR_VERSION}
