#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of nvdiffrast ${NVDIFFRAST_VERSION}"
	exit 1
fi

uv pip install nvdiffrast==${NVDIFFRAST_VERSION}
