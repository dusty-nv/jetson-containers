#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of cosmos1-diffusion-renderer ${COSMOS_DIFF_RENDER_VERSION}"
	exit 1
fi

pip3 install nvidia-cosmos==${COSMOS_DIFF_RENDER_VERSION}
