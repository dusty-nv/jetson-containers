#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of SpargeAttention ${SPARGE_ATTENTION_VERSION}"
	exit 1
fi

pip3 install spas_sage_attn==${SPARGE_ATTENTION_VERSION}
pip3 show spas_sage_attn && python3 -c 'from spas_sage_attn.autotune import (extract_sparse_attention_state_dict,load_sparse_attention_state_dict) '
