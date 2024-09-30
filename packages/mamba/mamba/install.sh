#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of mamba ${MAMBA}"
	exit 1
fi

pip3 install --no-cache-dir --verbose mamba_ssm==${MAMBA_VERSION}