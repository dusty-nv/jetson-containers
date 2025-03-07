#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of FlashAttention ${FLASH_ATTENTION_VERSION}"
	exit 1
fi

pip3 install flash-attn==${FLASH_ATTENTION_VERSION}
pip3 show flash-attn && python3 -c 'import flash_attn'
