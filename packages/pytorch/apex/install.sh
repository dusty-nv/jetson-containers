#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of apex ${APEX_VERSION}"
	exit 1
fi

pip3 install apex==${APEX_VERSION}
