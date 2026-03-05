#!/usr/bin/env bash
set -ex

echo "Installing OpenClaw ${OPENCLAW_VERSION}"

npm install -g npm@latest

if [ "$OPENCLAW_VERSION" = "latest" ]; then
    npm install -g openclaw@latest
else
    npm install -g openclaw@${OPENCLAW_VERSION}
fi

openclaw --version

echo "OpenClaw installed successfully"
