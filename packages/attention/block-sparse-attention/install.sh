#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of block_sparse_attn ${BLOCKSPARSEATTN_VERSION}"
	exit 1
fi

pip3 install block_sparse_attn==${BLOCKSPARSEATTN_VERSION}
