#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of bitsandbytes ${BITSANDBYTES_VERSION}"
	exit 1
fi

uv pip install scipy

# if it fails to install the specified version, try the dev version (like 0.45.4.dev0)
# which these wheels frequently get tagged as, but pip won't install as it evaluates as < than.
uv pip install bitsandbytes==${BITSANDBYTES_VERSION} || \
uv pip install --prerelease=allow "bitsandbytes>=${BITSANDBYTES_VERSION}.dev,<=${BITSANDBYTES_VERSION}"
