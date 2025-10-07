#!/usr/bin/env bash
# faster-whisper

set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of faster_whisper==${FASTER_WHISPER_VERSION}..."
	exit 1
fi

echo "Installing faster_whisper==${FASTER_WHISPER_VERSION}..."
uv pip install faster_whisper==${FASTER_WHISPER_VERSION}

uv pip show faster_whisper
python3 -c 'import faster_whisper; print(faster_whisper.__version__);'
