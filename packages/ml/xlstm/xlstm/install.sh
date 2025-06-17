#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of xlstm ${XLSTM_VERSION}"
	exit 1
fi

pip3 install xlstm==${XLSTM_VERSION} || \
pip3 install xlstm==${XLSTM_VERSION_SPEC}