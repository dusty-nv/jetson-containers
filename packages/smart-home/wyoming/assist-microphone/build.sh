#!/usr/bin/env bash
# wyoming-assist-microphone
set -euxo pipefail

apt-get update
apt-get install -y --no-install-recommends \
   netcat-traditional \
   libasound2-plugins \
   alsa-utils
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir -U \
   setuptools \
   wheel \
   webrtc-noise-gain==1.2.3 \
   pysilero-vad==1.0.0
pip3 install --no-cache-dir \
   "wyoming-satellite[webrtc] @ https://github.com/rhasspy/wyoming-satellite/archive/refs/tags/v${WYOMING_SATELLITE_VERSION}.tar.gz"

# Clone rootfs & config.aml
git clone --depth=1 https://github.com/home-assistant/addons /tmp/addons
git -C /tmp/addons sparse-checkout set --no-cone assist_microphone/ '!*/assist_microphone'

# Copy sounds
mkdir -p /usr/src/sounds
cp -r /tmp/addons/assist_microphone/sounds/* /usr/src/sounds/

rm -rf /tmp/addons
