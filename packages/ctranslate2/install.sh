#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of CTranslate2 ${CTRANSLATE_VERSION}"
	exit 1
fi

tarpack install ctranslate2-${CTRANSLATE_VERSION}
pip3 install --no-cache-dir --verbose ctranslate2==${CTRANSLATE_VERSION}
