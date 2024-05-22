#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of openai_triron ${OPENAITRIRON_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir --verbose openai_triron==${OPENAITRIRON_VERSION}
