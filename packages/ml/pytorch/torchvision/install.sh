#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchvision ${TORCHVISION_VERSION}"
	exit 1
fi

uv pip install torchvision~=${TORCHVISION_VERSION} || \
uv pip install --prerelease=allow "torchvision>=${TORCHVISION_VERSION}.dev,<=${TORCHVISION_VERSION}"

if [ "$(lsb_release -rs)" = "20.04" ]; then
    # https://github.com/conda/conda/issues/13619
    uv pip install pyopenssl==24.0.0
fi
