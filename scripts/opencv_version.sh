#!/usr/bin/env bash

# do this first:
# source scripts/docker_base.sh

echo "selecting OpenCV for L4T R$L4T_VERSION..."

if [[ $L4T_RELEASE -eq 32 ]]; then
	OPENCV_URL="https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz"
	OPENCV_DEB="OpenCV-4.5.0-aarch64.tar.gz"
elif [[ $L4T_RELEASE -eq 34 ]]; then
	OPENCV_URL="https://nvidia.box.com/shared/static/2hssa5g3v28ozvo3tc3qwxmn78yerca9.gz"
	OPENCV_DEB="OpenCV-4.5.0-aarch64.tar.gz"
fi

echo "OPENCV_URL=$OPENCV_URL"
echo "OPENCV_DEB=$OPENCV_DEB"
