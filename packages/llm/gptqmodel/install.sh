#!/usr/bin/env bash
set -ex

uv pip install gekko

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of AutoGPTQ ${AUTOGPTQ_VERSION}"
	exit 1
fi

uv pip install gptqmodel==${AUTOGPTQ_VERSION}

