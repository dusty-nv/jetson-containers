#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pykan ${PYKAN_VERSION}."
	exit 1
fi
uv pip install scikit-learn matplotlib pandas sympy pyyaml seaborn tqdm
uv pip install pykan==${PYKAN_VERSION} || \
uv pip install pykan==${PYKAN_VERSION_SPEC}
