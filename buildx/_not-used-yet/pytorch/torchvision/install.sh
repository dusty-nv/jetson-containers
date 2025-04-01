#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchvision ${TORCHVISION_VERSION}"
	exit 1
fi

pip3 install torchvision~=${TORCHVISION_VERSION}
   
if [ $(lsb_release --codename --short) = "focal" ]; then
    # https://github.com/conda/conda/issues/13619
    pip3 install pyopenssl==24.0.0
fi
