#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of cuda_cccl ${CCCL_VERSION}"
	exit 1
fi

uv pip install cuda-cccl==${CCCL_VERSION} || \
uv pip install cuda-cccl==${CCCL_VERSION_SPEC}

uv pip show cudnn && python3 -c 'import cudnn'
