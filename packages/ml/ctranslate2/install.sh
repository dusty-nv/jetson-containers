#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of CTranslate2 ${CTRANSLATE_VERSION}"
	exit 1
fi

tarpack install ctranslate2-${CTRANSLATE_VERSION}

uv pip install ctranslate2==${CTRANSLATE_VERSION} || \
uv pip install ctranslate2==4.5.0  # bump this to last released version
