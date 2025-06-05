#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pykan ${PYKAN_VERSION}."
	exit 1
fi
pip3 install scikit-learn matplotlib pandas sympy pyyaml seaborn tqdm
pip3 install pykan==${PYKAN_VERSION} || \
pip3 install pykan==${PYKAN_VERSION_SPEC}