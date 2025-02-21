#!/usr/bin/env bash
# psutil-home-assistant

set -euxo pipefail

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of psutil-home-assistant ${PSUTIL_HA_VERSION}..."
	exit 1
fi

echo "Installing psutil-home-assistant ${PSUTIL_HA_VERSION}..."

pip3 install --no-cache-dir --verbose psutil_home_assistant==${PSUTIL_HA_VERSION}

pip3 show psutil_home_assistant
python3 -c 'import psutil_home_assistant;'
