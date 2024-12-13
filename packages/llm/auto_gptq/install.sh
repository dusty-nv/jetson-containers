#!/usr/bin/env bash
set -ex

pip3 install --no-cache-dir gekko

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of AutoGPTQ ${AUTOGPTQ_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir --verbose auto-gptq==${AUTOGPTQ_VERSION}

