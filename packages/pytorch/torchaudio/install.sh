#!/usr/bin/env bash
# PyTorch installer
set -ex

# install prerequisites
uv pip install pysoundfile

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchaudio ${TORCHAUDIO_VERSION}"
	exit 1
fi

uv pip install torchaudio~=${TORCHAUDIO_VERSION} || \
uv pip install --prerelease=allow "torchaudio>=${TORCHAUDIO_VERSION}.dev,<=${TORCHAUDIO_VERSION}"
