#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of mamba ${MAMBA}"
	exit 1
fi

pip3 install mamba_ssm==${MAMBA_VERSION} || \
pip3 install mamba_ssm==${MAMBA_VERSION_SPEC}