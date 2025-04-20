#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pytorch3d ${PYTORCH3D_VERSION}"
	exit 1
fi

pip3 install pytorch3d~=${PYTORCH3D_VERSION}
