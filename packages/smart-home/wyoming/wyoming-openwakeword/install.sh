#!/usr/bin/env bash
# wyoming-openwakeword

set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of wyoming-openwakeword ${WYOMING_OPENWAKEWORD_VERSION}..."
	exit 1
fi

echo "Installing wyoming-openwakeword ${WYOMING_OPENWAKEWORD_VERSION}..."

uv pip install wyoming_openwakeword==${WYOMING_OPENWAKEWORD_VERSION}

uv pip show wyoming_openwakeword
python3 -c 'import wyoming_openwakeword; print(wyoming_openwakeword.__version__);'
