#!/usr/bin/env bash
set -euxo pipefail

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of cutlass ${CUTLASS_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir nvidia-cutlass==${CUTLASS_VERSION} pycute

# if #PYTHON_VERSION == "3.12" then install the DSL version
if [ "${PYTHON_VERSION}" == "3.12" ]; then
    echo "Installing nvidia-cutlass-dsl for Python 3.12"
    pip3 install nvidia-cutlass-dsl || echo "failed to install nvidia-cutlass-dsl for Python ${PYTHON_VERSION}"
else
    echo "Installing nvidia-cutlass for Python ${PYTHON_VERSION}"
fi
