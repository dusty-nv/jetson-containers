#!/usr/bin/env bash
# PyTorch installer
set -ex

# install prerequisites
pip3 install pysoundfile

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchaudio ${TORCHAUDIO_VERSION}"
	exit 1
fi

pip3 install torchaudio~=${TORCHAUDIO_VERSION} || \
pip3 install --pre "torchaudio>=${TORCHAUDIO_VERSION}.dev,<=${TORCHAUDIO_VERSION}"
