#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of fast_gauss ${4k4D_VERSION}"
	exit 1
fi

pip3 install fast_gauss==${4k4D_VERSION}