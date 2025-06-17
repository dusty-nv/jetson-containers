#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of cache_dit ${CACHE_DIT_VERSION}"
	exit 1
fi

pip3 install cache_dit==${CACHE_DIT_VERSION} || \
pip3 install cache_dit==${CACHE_DIT_VERSION_SPEC}
