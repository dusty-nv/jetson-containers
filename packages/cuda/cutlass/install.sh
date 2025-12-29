#!/usr/bin/env bash
set -euxo pipefail

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of cutlass ${CUTLASS_VERSION}"
	exit 1
fi

uv pip install --no-cache-dir nvidia-cutlass==${CUTLASS_VERSION} pycute

# if #PYTHON_VERSION == "3.12" then install the DSL version
echo "Installing nvidia-cutlass-dsl for Python 3.12"
uv pip install nvidia-cutlass-dsl || echo "failed to install nvidia-cutlass-dsl for Python ${PYTHON_VERSION}"
