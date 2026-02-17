#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchao ${TORCHAO_VERSION}"
	exit 1
fi

uv pip install torchao==${TORCHAO_VERSION} || \
uv pip install --prerelease=allow "torchao>=${TORCHAO_VERSION}.dev,<=${TORCHAO_VERSION}"
