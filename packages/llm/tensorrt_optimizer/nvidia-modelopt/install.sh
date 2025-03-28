#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of TensorRT-Model-Optimizer ${NVIDIA_MODELOPT_VERSION}"
	exit 1
fi

pip3 install nvidia-modelopt==${NVIDIA_MODELOPT_VERSION}