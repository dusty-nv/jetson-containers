#!/usr/bin/env bash
set -ex
uv pip install imageio-ffmpeg
if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of SageAttention ${SAGE_ATTENTION_VERSION}"
	exit 1
fi

uv pip install sageattention==${SAGE_ATTENTION_VERSION}
uv pip show sageattention && python3 -c 'from sageattention import sageattn'
