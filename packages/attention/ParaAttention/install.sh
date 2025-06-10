#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of ParaAttention ${PARAATENTTION_VERSION}"
	exit 1
fi

pip3 install para-attn==${PARAATENTTION_VERSION}
