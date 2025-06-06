#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of MemVid ${MEMVID_VERSION}"
	exit 1
fi

pip3 install PyPDF2
pip3 install memvid==${MEMVID_VERSION}
