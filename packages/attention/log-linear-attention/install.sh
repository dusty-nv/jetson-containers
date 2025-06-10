#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of log-linear-attention ${LOG_LINEAR_ATTN_VERSION}"
	exit 1
fi

pip3 install hattention==${LOG_LINEAR_ATTN_VERSION}
