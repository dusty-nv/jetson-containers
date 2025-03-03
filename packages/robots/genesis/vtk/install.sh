#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of vtk ${VTK}"
	exit 1
fi

pip3 install vtk==${VTK_VERSION}