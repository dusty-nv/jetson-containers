#!/usr/bin/env bash
# ciso8601
set -ex

apt-get update
apt-get install -y --no-install-recommends \
        autoconf \
        libpcap0.8
apt-get clean
rm -rf /var/lib/apt/lists/*

git clone https://github.com/closeio/ciso8601 /tmp/ciso8601
git -C /tmp/ciso8601 apply /tmp/ciso8601-patch.diff
git -C /tmp/ciso8601 diff

pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/ciso8601
pip3 install --no-cache-dir --verbose /opt/ciso8601*.whl
pip3 show ciso8601

python3 -c 'import ciso8601; print(ciso8601.__version__);'
rm -rf /tmp/ciso8601