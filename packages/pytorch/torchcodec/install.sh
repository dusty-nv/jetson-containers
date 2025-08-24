#!/usr/bin/env bash
# PyTorch installer
set -ex

# install prerequisites
pip3 install pysoundfile

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchcodec ${TORCHCODEC_VERSION}"
	exit 1
fi

pip3 install torchcodec~=${TORCHCODEC_VERSION} || \
pip3 install --pre "torchcodec>=${TORCHCODEC_VERSION}.dev,<=${TORCHCODEC_VERSION}"
