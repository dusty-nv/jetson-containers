#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of polyscope ${POLYSCOPE_VERSION}"
	exit 1
fi

pip3 install polyscope==${POLYSCOPE_VERSION} || \
pip3 install polyscope==${POLYSCOPE_VERSION_SPEC}
