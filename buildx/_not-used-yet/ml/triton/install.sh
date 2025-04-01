#!/usr/bin/env bash
#triton
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of triton ${TRITON_VERSION} (branch=${TRITON_BRANCH})"
	exit 1
fi

pip3 install triton==${TRITON_VERSION}
