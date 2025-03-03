#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of FlashInfer ${FLASHINFER_VERSION}"
	exit 1
fi

pip3 install flashinfer==${FLASHINFER_VERSION}
pip3 show flashinfer && python3 -c 'import flashinfer'
