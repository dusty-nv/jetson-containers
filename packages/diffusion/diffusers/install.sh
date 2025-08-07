#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of diffusers ${DIFFUSERS_VERSION}"
	exit 1
fi

pip3 install diffusers==${DIFFUSERS_VERSION} || \
	pip3 install diffusers==${DIFFUSERS_VERSION}.dev0
