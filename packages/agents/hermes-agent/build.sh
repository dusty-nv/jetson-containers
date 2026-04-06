#!/usr/bin/env bash
set -ex

echo "Installing Hermes Agent ${HERMES_AGENT_VERSION}"

npm install -g npm@latest

REPO_URL="https://github.com/NousResearch/hermes-agent.git"

if [ "$HERMES_AGENT_VERSION" = "latest" ]; then
    git clone --depth 1 ${REPO_URL} /tmp/hermes-agent/src
else
    git clone --depth 1 --branch v${HERMES_AGENT_VERSION} ${REPO_URL} /tmp/hermes-agent/src
fi

cd /tmp/hermes-agent/src

pip3 install --no-cache-dir uv
uv pip install --system -e ".[all]"
npm install

hermes --version

rm -rf /tmp/hermes-agent/src

echo "Hermes Agent installed successfully"
