#!/usr/bin/env bash
set -ex

echo "Building OpenFang ${OPENFANG_VERSION}"

REPO_URL="https://github.com/RightNow-AI/openfang.git"

if [ "$OPENFANG_VERSION" = "latest" ]; then
    git clone --depth 1 ${REPO_URL} /tmp/openfang/src
else
    git clone --depth 1 --branch v${OPENFANG_VERSION} ${REPO_URL} /tmp/openfang/src
fi

cd /tmp/openfang/src

cargo build --workspace --release

cp target/release/openfang /usr/local/bin/openfang
chmod +x /usr/local/bin/openfang

openfang --version

rm -rf /tmp/openfang/src

echo "OpenFang installed successfully"
