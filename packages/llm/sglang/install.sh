#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of sglang ${SGLANG_VERSION}"
	exit 1
fi

uv pip3 install sglang==${SGLANG_VERSION} || \
uv pip install sglang==${SGLANG_VERSION_SPEC}