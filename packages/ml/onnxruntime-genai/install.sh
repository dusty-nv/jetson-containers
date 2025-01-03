#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of onnxruntime-genai ${ONNXRUNTIME_GENAI_VERSION} (branch=${ONNXRUNTIME_GENAI_BRANCH}"
	exit 1
fi

pip3 install --no-cache-dir --verbose onnxruntime-genai==${ONNXRUNTIME_GENAI_VERSION}

python3 -c 'import onnxruntime-genai; print(onnxruntime-genai.__version__);'