#!/usr/bin/env bash
# wyoming-whisper

set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of wyoming-whisper ${WYOMING_WHISPER_VERSION}..."
	exit 1
fi

echo "Installing wyoming-whisper ${WYOMING_WHISPER_VERSION}..."

pip3 install --no-cache-dir --verbose wyoming_faster_whisper==${WYOMING_WHISPER_VERSION}

pip3 show wyoming_faster_whisper
python3 -c 'import wyoming_faster_whisper; print(wyoming_faster_whisper.__version__);'
