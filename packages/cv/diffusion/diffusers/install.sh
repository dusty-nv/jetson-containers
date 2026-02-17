#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of diffusers ${DIFFUSERS_VERSION}"
	exit 1
fi

uv pip install diffusers==${DIFFUSERS_VERSION} || \
	uv pip install diffusers==${DIFFUSERS_VERSION}.dev0
