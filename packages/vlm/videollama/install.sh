#!/usr/bin/env bash
set -ex

uv pip install "transformers<=4.52"
if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of videollama ${VIDEOLLAMA_VERSION}"
	exit 1
fi

uv pip install videollama==${VIDEOLLAMA_VERSION}
