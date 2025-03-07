#!/usr/bin/env bash
set -ex

pip3 install gekko

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of AutoGPTQ ${AUTOGPTQ_VERSION}"
	exit 1
fi

pip3 install auto-gptq==${AUTOGPTQ_VERSION}

