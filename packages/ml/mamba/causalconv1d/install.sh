#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of causal-conv1d ${CAUSALCONV1D}"
	exit 1
fi

uv pip install causal_conv1d==${CASUALCONV1D_VERSION} || \
uv pip install causal_conv1d==${CASUALCONV1D_VERSION_SPEC}
