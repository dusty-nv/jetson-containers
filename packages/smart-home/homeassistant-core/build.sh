#!/usr/bin/env bash
# homeassistant-core
set -euxo pipefail

echo "Installing Home Assistant Core ${HA_VERSION}..."

apt-get update
apt-get install -y --no-install-recommends \
        autoconf \
        libpcap0.8 \
        libturbojpeg
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir --ignore-installed blinker
pip3 install --no-cache-dir --verbose uv==0.1.27 ruff

# Install homeassistant-core
git clone --branch=${HA_VERSION} https://github.com/home-assistant/core /usr/src/homeassistant
uv pip install --no-cache-dir --verbose \
        -r /usr/src/homeassistant/requirements.txt \
        -r /usr/src/homeassistant/requirements_all.txt
uv pip install -e /usr/src/homeassistant
python3 -m compileall /usr/src/homeassistant

# Generate languages
cd /usr/src/homeassistant
python3 -m script.hassfest
python3 -m script.translations develop --all

# Copy Home Assistant S6-Overlay
cp -r rootfs/* /