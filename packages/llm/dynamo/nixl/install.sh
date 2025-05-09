#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of nixl ${NIXL_VERSION}"
	exit 1
fi
"ai-nixl[all]"

pip3 install "nixl~=${NIXL_VERSION}" || \
pip3 install "nixl~=${NIXL_VERSION_SPEC}"