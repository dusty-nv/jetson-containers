#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torch_memory_saver ${TORCH_MEMORY_SAVER_VERSION}"
	exit 1
fi

uv pip install torch-memory-saver==${TORCH_MEMORY_SAVER_VERSION} || \
uv pip install --pre "torch-memory-saver>=${TORCH_MEMORY_SAVER_VERSION}.dev,<=${TORCH_MEMORY_SAVER_VERSION}"
