#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of cache_dit ${CACHE_DIT_VERSION}"
	exit 1
fi

uv pip install cache_dit==${CACHE_DIT_VERSION} || \
uv pip install cache_dit==${CACHE_DIT_VERSION_SPEC}
