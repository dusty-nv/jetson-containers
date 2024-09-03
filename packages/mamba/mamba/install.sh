#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of mamba ${HLOC}"
	exit 1
fi

pip3 install --no-cache-dir --verbose mamba==${MAMBA_VERSION}