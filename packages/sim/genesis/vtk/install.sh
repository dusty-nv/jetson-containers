#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of vtk ${VTK}"
	exit 1
fi

uv pip install vtk==${VTK_VERSION} || \
uv pip install vtk==${VTK_VERSION_SPEC}
