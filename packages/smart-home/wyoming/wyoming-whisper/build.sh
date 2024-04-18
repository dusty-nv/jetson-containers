#!/usr/bin/env bash
# wyoming-whisper
set -euxo pipefail

apt-get update
apt-get install -y --no-install-recommends \
   netcat
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir -U \
   setuptools \
   wheel

# Clone wyoming-faster-whisper layer
git clone https://github.com/rhasspy/wyoming-faster-whisper /tmp/wyoming-faster-whisper
pip3 install --no-cache-dir --verbose -r /tmp/wyoming-faster-whisper/requirements.txt
pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/wyoming-faster-whisper
pip3 install --no-cache-dir --verbose /opt/wyoming_faster_whisper*.whl
pip3 show wyoming_faster_whisper
rm -rf /tmp/wyoming-faster-whisper

# Clone rootfs
git clone --depth=1 https://github.com/home-assistant/addons /tmp/addons
git -C /tmp/addons sparse-checkout set --no-cone whisper/ '!*/whisper'

# Fix: Adjust reading config data from config.yaml file when not using HA Supervisor
sed -i \
   -e "s/bashio::config.true 'debug_logging'/[[ \"$WHISPER_DEBUG\" == \"true\" ]]/g" \
   -e "s/model=\".*/model=\"${WHISPER_MODEL}\"/g" \
   -e "s/--beam-size.*/--beam-size \"$WHISPER_BEAM_SIZE\" \\\\/g" \
   -e "s/--language.*/--language \"$WHISPER_LANGUAGE\" \\\\/g" \
   -e "s/--initial-prompt.*/--initial-prompt \"$WHISPER_INITIAL_PROMPT\" \\\\/g" \
   /tmp/addons/whisper/rootfs/etc/s6-overlay/s6-rc.d/whisper/run

# Fix: Disable native Discovery handled by HA Supervisor
cat /tmp/addons/whisper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/whisper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/whisper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/whisper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/whisper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/whisper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
cat /tmp/addons/whisper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run

cp -r /tmp/addons/whisper/rootfs/* /
cp /tmp/addons/whisper/config.yaml /etc/s6-overlay/s6-rc.d/whisper/config.yaml
rm -rf /tmp/addons

python3 -c 'import wyoming_faster_whisper; print(wyoming_faster_whisper.__version__);'
