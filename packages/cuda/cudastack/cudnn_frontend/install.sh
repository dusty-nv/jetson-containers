#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of cudnn_frontend ${CUDNN_FRONTEND_VERSION}"
	exit 1
fi

uv pip install nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION} || \
uv pip install nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION_SPEC}

pip3 show nvidia-cudnn-frontend && python3 -c 'import cudnn'
