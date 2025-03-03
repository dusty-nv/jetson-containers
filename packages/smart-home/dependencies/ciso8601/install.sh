#!/usr/bin/env bash
# ciso8601

set -euxo pipefail

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of ciso8601 ${CISO8601_VERSION}..."
	exit 1
fi

echo "Installing ciso8601 ${CISO8601_VERSION}..."

pip3 install ciso8601==${CISO8601_VERSION}

pip3 show ciso8601
python3 -c 'import ciso8601; print(ciso8601.__version__);'
