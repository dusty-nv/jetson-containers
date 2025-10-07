#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of plstm ${PLSTM_VERSION}"
	exit 1
fi

uv pip install plstm==${PLSTM_VERSION} || \
uv pip install plstm==${PLSTM_VERSION_SPEC}
