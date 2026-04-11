#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of multi-turboquant ${MULTI_TURBOQUANT_VERSION}"
	exit 1
fi

uv pip install multi-turboquant==${MULTI_TURBOQUANT_VERSION}
