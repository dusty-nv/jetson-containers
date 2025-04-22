#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of SageAttention ${SAGE_ATTENTION_VERSION}"
	exit 1
fi

pip3 install sageattention==${SAGE_ATTENTION_VERSION}
pip3 show sageattention && python3 -c 'from sageattention import sageattn'
