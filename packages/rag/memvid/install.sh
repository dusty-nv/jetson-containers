#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of MemVid ${MEMVID_VERSION}"
	exit 1
fi

uv pip install PyPDF2
uv pip install memvid==${MEMVID_VERSION}
