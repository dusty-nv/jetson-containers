#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of 3DGRUT ${3DGRUT_VERSION}"
	exit 1
fi

pip3 install threedgrut==${3DGRUT_VERSION}