#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of FlashInfer ${FLASHINFER_VERSION}"
	exit 1
fi

uv pip install flashinfer-python==${FLASHINFER_VERSION} || \
uv pip install flashinfer-python==${FLASHINFER_VERSION_SPEC}

uv pip show flashinfer_python && python3 -c 'import flashinfer'
