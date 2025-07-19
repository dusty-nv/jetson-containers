#!/usr/bin/env bash
set -ex

if [ "on" == "on" ]; then
	echo "Forcing build of cudnn_frontend ${CUDNN_FRONTEND_VERSION}"
	exit 1
fi

pip3 install nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION} || \
pip3 install nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION_SPEC}

pip3 show cudnn && python3 -c 'import cudnn'
