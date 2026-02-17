#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of nerfstudio ${NERFSTUDIO}"
	exit 1
fi

uv pip install --no-deps --no-build-isolation nerfstudio #==${NERFSTUDIO_VERSION}
uv pip install -U --force-reinstall opencv-python-contrib
