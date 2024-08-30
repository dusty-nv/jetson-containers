#!/usr/bin/env bash
# PyTorch installer
set -ex

# install prerequisites
pip3 install --no-cache-dir --verbose pysoundfile

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchaudio ${TORCHAUDIO_VERSION}"
	exit 1
fi

pip3 install --no-cache-dir torchaudio==${TORCHAUDIO_VERSION}
mkdir -p /torch-wheels
pip3 download --verbose --no-deps -d /torch-wheels torchaudio==${TORCHAUDIO_VERSION}
   
