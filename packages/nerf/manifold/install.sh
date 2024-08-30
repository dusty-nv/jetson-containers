#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of manifold ${MANIFOLD}"
	exit 1
fi

pip3 install --no-cache-dir --verbose manifold3d==${MANIFOLD_VERSION}