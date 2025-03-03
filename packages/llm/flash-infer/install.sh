#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of FlashInfer ${FLASHINFER_VERSION}"
	exit 1
fi

pip3 install flashinfer-python==${FLASHINFER_VERSION} || \
pip3 install flashinfer-python==${FLASHINFER_VERSION_SPEC}

pip3 show flashinfer_python && python3 -c 'import flashinfer'
