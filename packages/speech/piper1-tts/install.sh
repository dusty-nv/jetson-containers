#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of piper1-tts ${PIPER_VERSION} (${PIPER_BRANCH})"
	exit 1
fi

# install the wheel
pip3 install piper==${PIPER_VERSION}

# make sure it loads
pip3 show piper
