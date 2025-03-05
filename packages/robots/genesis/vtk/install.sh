#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of vtk ${VTK}"
	exit 1
fi

pip3 install sglang==${VTK_VERSION} || \
pip3 install sglang==${VTK_VERSION_SPEC}