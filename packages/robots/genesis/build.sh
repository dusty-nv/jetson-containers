#!/usr/bin/env bash
set -ex

echo "Setting up environment for genesis ${GENESIS_VERSION}"

echo "Building genesis ${GENESIS_VERSION}"
# Clone the repository if it doesn't exist
git clone --branch=v${GENESIS_VERSION} --depth=1 --recursive https://github.com/Genesis-Embodied-AI/Genesis /opt/genesis || \
git clone --depth=1 --recursive https://github.com/Genesis-Embodied-AI/Genesis /opt/genesis

cd /opt/genesis

pip3 wheel . -w /opt/genesis/wheels

pip3 install --no-cache-dir --verbose /opt/genesis/wheels/genesis-world*.whl

twine upload --verbose /opt/genesis/wheels/genesis-world*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

echo "genesis OK\n"

