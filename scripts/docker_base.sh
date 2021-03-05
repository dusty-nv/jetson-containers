#!/usr/bin/env bash

source scripts/l4t_version.sh

BASE_IMAGE="nvcr.io/nvidia/l4t-base:r$L4T_VERSION"
BASE_DEVEL="nvcr.io/nvidian/nvidia-l4t-base:r$L4T_VERSION"

if [ $L4T_RELEASE -eq 32 ]; then
	if [ $L4T_REVISION_MAJOR -eq 4 ]; then
		if [ $L4T_REVISION_MINOR -gt 4 ]; then
			BASE_IMAGE=$BASE_DEVEL
		fi
	elif [ $L4T_REVISION_MAJOR -eq 5 ]; then
		if [ $L4T_REVISION_MINOR -eq 1 ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-base:r32.5.0"
		elif [ $L4T_REVISION_MINOR -gt 1 ]; then
			BASE_IMAGE=$BASE_DEVEL
		fi
	elif [ $L4T_REVISION_MAJOR -gt 5 ]; then
		BASE_IMAGE=$BASE_DEVEL
	fi
fi
	
echo "l4t-base image:  $BASE_IMAGE"