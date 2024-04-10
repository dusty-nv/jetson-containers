#!/usr/bin/env bash
# wyoming-openwakeword
set -ex

apt-get update
apt-get install -y --no-install-recommends \
   netcat-traditional \
   libopenblas0
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir -U \
   setuptools \
   wheel
   
echo "wyoming-openwakeword: ${OPENWAKEWORD_VERSION}"

pip3 install --no-cache-dir \
   --extra-index-url https://www.piwheels.org/simple \
   "wyoming-openwakeword @ https://github.com/rhasspy/wyoming-openwakeword/archive/refs/tags/v${OPENWAKEWORD_VERSION}.tar.gz"
   
git clone --depth=1 https://github.com/home-assistant/addons /tmp/addons
git -C /tmp/addons sparse-checkout set --no-cone openwakeword/ '!*/openwakeword'

# Fix: Adjust reading config data from config.yaml file when not using HA Supervisor
sed -i \
   -e "s/bashio::config.true 'debug_logging'/[[ \"$OPENWAKEWORD_DEBUG\" == \"true\" ]]/g" \
   -e "s/--threshold.*/--threshold \"$OPENWAKEWORD_THRESHOLD\" \\\\/g" \
   -e "s/--trigger-level.*/--trigger-level \"$OPENWAKEWORD_TRIGGER_LEVEL\" \${flags[@]}/g" \
   /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/openwakeword/run

# Fix: Disable native Discovery handled by HA Supervisor 
cat /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
cat /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run

cp -r /tmp/addons/openwakeword/rootfs/* /
cp /tmp/addons/openwakeword/config.yaml /etc/s6-overlay/s6-rc.d/openwakeword/config.yaml
rm -rf /tmp/addons

python3 -c 'import wyoming_openwakeword; print(wyoming_openwakeword.__version__);'
