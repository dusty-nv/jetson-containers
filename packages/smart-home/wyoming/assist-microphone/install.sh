#!/usr/bin/env bash
# wyoming-assist-microphone
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of wyoming-assist-microphone ${SATELLITE_VERSION}..."
	exit 1
fi

echo "Installing wyoming-assist-microphone ${SATELLITE_VERSION}..."

pip3 install --no-cache-dir --verbose assist_microphone[silerovad,webrtc]==${SATELLITE_VERSION}

pip3 show assist_microphone
python3 -c 'import assist_microphone; print(assist_microphone.__version__);'
