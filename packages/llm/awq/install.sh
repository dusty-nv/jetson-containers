#!/usr/bin/env bash
set -ex

uv pip install 'lm-eval<=0.3.0'

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of AWQ ${AWQ_VERSION} (kernels=${AWQ_KERNEL_VERSION})"
	exit 1
fi

uv pip install awq==${AWQ_VERSION} awq-inference-engine==${AWQ_KERNEL_VERSION}
