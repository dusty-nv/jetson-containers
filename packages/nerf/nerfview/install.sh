#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of nerfview ${NERFVIEW_VERSION}"
	exit 1
fi

pip3 install nerfview==${NERFVIEW_VERSION}