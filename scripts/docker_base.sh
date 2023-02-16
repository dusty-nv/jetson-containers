#!/usr/bin/env bash

source scripts/l4t_version.sh

if [ $ARCH = "aarch64" ]; then
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
			elif [ $L4T_REVISION_MINOR -eq 2 ]; then
				BASE_IMAGE="nvcr.io/nvidia/l4t-base:r32.5.0"
			elif [ $L4T_REVISION_MINOR -gt 2 ]; then
				BASE_IMAGE=$BASE_DEVEL
			fi
		elif [ $L4T_REVISION_MAJOR -gt 7 ]; then
			BASE_IMAGE=$BASE_DEVEL
		fi
		
	elif [ $L4T_RELEASE -eq 34 ]; then # JetPack 5.0.0 (DP) / 5.0.1 (DP2)
		if [[ $L4T_REVISION_MAJOR -eq 1 && $L4T_REVISION_MINOR -eq 0 ]]; then
			BASE_DEVEL="nvcr.io/nvidian/nvidia-l4t-base:r34.1"
			BASE_IMAGE="nvcr.io/nvidia/l4t-base:r34.1"
		fi
		
		BASE_IMAGE_L4T=$BASE_IMAGE
		BASE_IMAGE="jetpack:r$L4T_VERSION"
		
	elif [ $L4T_RELEASE -eq 35 ]; then
		if [ $L4T_REVISION_MAJOR -le 1 ]; then # JetPack 5.0.2 (GA)
			BASE_IMAGE_L4T="nvcr.io/nvidia/l4t-base:r35.1.0"
			BASE_IMAGE="nvcr.io/nvidia/l4t-jetpack:r35.1.0"
		else  # JetPack 5.1
			BASE_IMAGE_L4T="nvcr.io/nvidia/l4t-base:r35.2.1"
			BASE_IMAGE="nvcr.io/nvidia/l4t-jetpack:r35.2.1"
		fi
	fi

elif [ $ARCH = "x86_64" ]; then
	BASE_IMAGE="nvcr.io/nvidia/pytorch:22.04-py3"
fi

echo "L4T Base Image:   $BASE_IMAGE"