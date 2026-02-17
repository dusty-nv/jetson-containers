#!/usr/bin/env bash
# PyTorch installer
set -ex

# install prerequisites
uv pip install pysoundfile

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchcodec ${TORCHCODEC_VERSION}"
	exit 1
fi

uv pip install torchcodec~=${TORCHCODEC_VERSION} || \
uv pip install --prerelease=allow "torchcodec>=${TORCHCODEC_VERSION}.dev,<=${TORCHCODEC_VERSION}"
