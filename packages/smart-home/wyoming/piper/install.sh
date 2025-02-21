#!/usr/bin/env bash
# wyoming-piper

set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of wyoming-piper ${WYOMING_PIPER_VERSION}..."
	exit 1
fi

echo "Installing wyoming-piper ${WYOMING_PIPER_VERSION}..."

pip3 install --no-cache-dir --verbose wyoming_piper==${WYOMING_PIPER_VERSION}

pip3 show wyoming_piper
python3 -c 'import wyoming_piper; print(wyoming_piper.__version__);'
