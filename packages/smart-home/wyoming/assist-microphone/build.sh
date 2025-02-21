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

pip3 install --no-cache-dir -U \
   setuptools \
   wheel \
   webrtc-noise-gain==1.2.3 \
   pysilero-vad==1.0.0

echo "assist_microphone: ${SATELLITE_VERSION} (branch: ${SATELLITE_BRANCH})"

git clone --branch=${SATELLITE_BRANCH} https://github.com/rhasspy/wyoming-satellite /tmp/wyoming_satellite
cd /tmp/wyoming_satellite

sed -i "s|version=\"[^\"]*\"|version=\"${SATELLITE_VERSION}\"|" setup.py

python3 setup.py sdist bdist_wheel --verbose --dist-dir /opt/wheels

cd /
rm -rf /tmp/wyoming_satellite

pip3 install --no-cache-dir /opt/wheels/wyoming_satellite*.whl

pip3 show wyoming_satellite
python3 -c 'import wyoming_satellite; print(wyoming_satellite.__version__);'

twine upload --skip-existing --verbose /opt/wheels/wyoming_satellite*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm /opt/wheels/wyoming_satellite*.whl

# Clone rootfs & config.aml
git clone --depth=1 https://github.com/home-assistant/addons /tmp/addons
git -C /tmp/addons sparse-checkout set --no-cone assist_microphone/ '!*/assist_microphone'

# Copy sounds
mkdir -p /usr/src/sounds
cp -r /tmp/addons/assist_microphone/sounds/* /usr/src/sounds/

rm -rf /tmp/addons
