#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of FlexPrefill ${FLEXPREFILL_VERSION}"
	exit 1
fi

pip3 install flex_prefill==${FLEXPREFILL_VERSION}
