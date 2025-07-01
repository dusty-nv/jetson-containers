#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of RadialAttention ${RADIAL_ATTENTION_VERSION}"
	exit 1
fi

pip3 install radial-attention==${RADIAL_ATTENTION_VERSION}
