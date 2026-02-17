#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of polyscope ${POLYSCOPE_VERSION}"
	exit 1
fi

uv pip install polyscope==${POLYSCOPE_VERSION} || \
uv pip install polyscope==${POLYSCOPE_VERSION_SPEC}
