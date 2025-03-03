#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of manifold ${MANIFOLD}"
	exit 1
fi

pip3 install manifold3d==${MANIFOLD_VERSION}