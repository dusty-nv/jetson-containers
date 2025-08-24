#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of FlashAttention ${TILELANG_VERSION}"
	exit 1
fi

pip3 install tilelang==${TILELANG_VERSION}
pip3 show tilelang && python3 -c 'import tilelang'
