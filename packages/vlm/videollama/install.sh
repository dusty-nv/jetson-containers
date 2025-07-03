#!/usr/bin/env bash
set -ex

pip3 install "transformers<=4.52"
if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of videollama ${VIDEOLLAMA_VERSION}"
	exit 1
fi

pip3 install videollama==${VIDEOLLAMA_VERSION}
