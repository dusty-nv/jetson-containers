#!/usr/bin/env bash
set -ex

echo "Building ZeroClaw ${ZEROCLAW_VERSION}"

REPO_URL="https://github.com/zeroclaw-labs/zeroclaw.git"

if [ "$ZEROCLAW_VERSION" = "latest" ]; then
    git clone --depth 1 ${REPO_URL} /tmp/zeroclaw/src
else
    git clone --depth 1 --branch v${ZEROCLAW_VERSION} ${REPO_URL} /tmp/zeroclaw/src
fi

cd /tmp/zeroclaw/src

cargo build --release --locked

cp target/release/zeroclaw /usr/local/bin/zeroclaw
chmod +x /usr/local/bin/zeroclaw

zeroclaw --version

rm -rf /tmp/zeroclaw/src

echo "ZeroClaw installed successfully"
