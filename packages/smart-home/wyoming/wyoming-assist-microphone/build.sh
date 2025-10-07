#!/usr/bin/env bash
# wyoming-assist-microphone
set -ex

apt-get update
apt-get install -y --no-install-recommends --fix-missing \
   netcat-traditional \
   libasound2-plugins \
   alsa-utils
apt-get clean
rm -rf /var/lib/apt/lists/*

uv pip install -U \
   build \
   setuptools \
   wheel \
   webrtc-noise-gain==1.2.3 \
   pysilero-vad==1.0.0

git clone --branch=${SATELLITE_BRANCH} https://github.com/rhasspy/wyoming-satellite /tmp/wyoming_satellite
cd /tmp/wyoming_satellite

python3 -m build --wheel --sdist --outdir $PIP_WHEEL_DIR

cd /
rm -rf /tmp/wyoming_satellite

uv pip install $PIP_WHEEL_DIR/wyoming_satellite*.whl

uv pip show wyoming_satellite

twine upload --verbose $PIP_WHEEL_DIR/wyoming_satellite*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/wyoming_satellite*.whl

# Clone rootfs & config.aml
git clone --depth=1 https://github.com/home-assistant/addons /tmp/addons
git -C /tmp/addons sparse-checkout set --no-cone assist_microphone/ '!*/assist_microphone'

# Copy sounds
mkdir -p /usr/src/sounds
cp -r /tmp/addons/assist_microphone/sounds/* /usr/src/sounds/

rm -rf /tmp/addons
