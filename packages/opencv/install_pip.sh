#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of opencv-python ${OPENCV_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir --verbose opencv-contrib-python~=${OPENCV_VERSION}

python3 -c 'import cv2; print(cv2.getBuildInformation());'