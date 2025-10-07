#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pyrealsense2 ${LIBREALSENSE_VERSION}"
	exit 1
fi

pip3 install pyrealsense2==${LIBREALSENSE_VERSION} || \
pip3 install pyrealsense2==${LIBREALSENSE_VERSION_SPEC}