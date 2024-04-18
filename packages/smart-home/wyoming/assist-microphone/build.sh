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
   wheel
pip3 install --no-cache-dir \
   "wyoming-satellite[webrtc] @ https://github.com/rhasspy/wyoming-satellite/archive/refs/tags/v${WYOMING_SATELLITE_VERSION}.tar.gz"

# Clone rootfs
git clone --depth=1 https://github.com/home-assistant/addons /tmp/addons
git -C /tmp/addons sparse-checkout set --no-cone assist_microphone/ '!*/assist_microphone'

# Fix: Adjust reading config data from config.yaml file when not using HA Supervisor
sed -i \
   -e "s#bashio::config.true 'debug_logging'#[[ \"$ASSIST_MICROPHONE_DEBUG\" == \"true\" ]]#g" \
   -e "s#bashio::config.true 'sound_enabled'#[[ \"$ASSIST_MICROPHONE_SOUND_ENABLED\" == \"true\" ]]#g" \
   -e "s#--awake-wav.*#--awake-wav \"$ASSIST_MICROPHONE_AWAKE_WAV\" \\\#g" \
   -e "s#--done-wav.*#--done-wav \"$ASSIST_MICROPHONE_DONE_WAV\" \\\#g" \
   -e "s#--mic-volume-multiplier.*#--mic-volume-multiplier $ASSIST_MICROPHONE_MIC_VOLUME_MULTIPLIER \\\#g" \
   -e "s#--snd-volume-multiplier.*#--snd-volume-multiplier $ASSIST_MICROPHONE_SND_VOLUME_MULTIPLIER \\\#g" \
   -e "s#--mic-auto-gain.*#--mic-auto-gain $ASSIST_MICROPHONE_MIC_AUTO_GAIN \\\#g" \
   -e "s#--mic-noise-suppression.*#--mic-noise-suppression $ASSIST_MICROPHONE_MIC_NOISE_SUPPRESSION \\\#g" \
   /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/assist_microphone/run

# TODO: Add enviroment variables to handle openWakeWord integration
sed -i '25a\    --wake-uri "tcp://127.0.0.1:10400" \\' /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/assist_microphone/run
sed -i '25a\    --wake-word-name "ok_nabu" \\' /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/assist_microphone/run
cat /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/assist_microphone/run

# TODO: Create common bash scripts for homeassistant related containers
# Fix: Disable native Discovery handled by HA Supervisor
sed -i '$d' /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
cat /tmp/addons/assist_microphone/rootfs/etc/s6-overlay/s6-rc.d/discovery/run

# Copy modiefied rootfs
cp -r /tmp/addons/assist_microphone/rootfs/* /
cp /tmp/addons/assist_microphone/config.yaml /etc/s6-overlay/s6-rc.d/assist_microphone/config.yaml

# Copy sounds
mkdir -p /usr/src/sounds
cp -r /tmp/addons/assist_microphone/sounds/* /usr/src/sounds/

rm -rf /tmp/addons
