#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of onnxruntime_genai ${ONNXRUNTIME_GENAI_VERSION} (branch=${ONNXRUNTIME_GENAI_BRANCH}"
	exit 1
fi

pip3 install --no-cache-dir --verbose onnxruntime_genai==${ONNXRUNTIME_GENAI_VERSION}

python3 -c 'import onnxruntime_genai; print(onnxruntime_genai.__version__);'