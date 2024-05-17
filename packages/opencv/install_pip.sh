#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of opencv-python ${OPENCV_VERSION}"
	exit 1
fi

if [ ! -z "$OPENCV_URL" ]; then
    echo "Installing opencv from deb packages"
    exit 1
fi

pip3 install --no-cache-dir --verbose opencv-contrib-python~=${OPENCV_VERSION}
