#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of dynamo ${DYNAMO_VERSION}"
	exit 1
fi

pip3 install ai-dynamo==${DYNAMO_VERSION}