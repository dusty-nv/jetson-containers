#!/usr/bin/env bash
set -ex

echo "Building MemVid ${MEMVID_VERSION}"

git clone --depth=1 --branch=v${MEMVID_VERSION} https://github.com/Olow304/memvid /opt/memvid ||
git clone --depth=1 https://github.com/Olow304/memvid /opt/memvid

cd /opt/memvid

sed -i 's/==/>=/g' requirements.txt
uv pip install -U -r requirements.txt
uv build --wheel . -v --no-deps --out-dir /opt/memvid/wheels/
uv pip install /opt/memvid/wheels/memvid*.whl
twine upload --verbose /opt/memvid/wheels/memvid*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
