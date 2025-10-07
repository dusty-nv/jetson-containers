#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of transformer_engine ${TRANSFORMER_ENGINE}"
	exit 1
fi

uv pip install transformer-engine==${TRANSFORMER_ENGINE_VERSION}
