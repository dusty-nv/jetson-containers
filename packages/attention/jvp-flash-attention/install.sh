#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of JVP-FlashAttention ${JVP_FLASH_ATTENTION_VERSION}"
	exit 1
fi

uv pip install jvp-flash-attention==${JVP_FLASH_ATTENTION_VERSION}
uv pip show jvp-flash-attention && python3 -c 'import flash_attn'
