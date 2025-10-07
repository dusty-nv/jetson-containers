#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of onnxruntime ${ONNXRUNTIME_VERSION} (branch=${ONNXRUNTIME_BRANCH}, flags=${ONNXRUNTIME_FLAGS})"
	exit 1
fi

tarpack install onnxruntime-gpu-${ONNXRUNTIME_VERSION}
uv pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}

python3 -c 'import onnxruntime; print(onnxruntime.__version__);'
