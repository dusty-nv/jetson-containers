#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of open3d ${OPEN3D_VERSION}"
	exit 1
fi

pip3 install open3d==${OPEN3D_VERSION}
