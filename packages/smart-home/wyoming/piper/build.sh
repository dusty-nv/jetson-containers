#!/usr/bin/env bash
# wyoming-piper
set -euxo pipefail

apt-get update
apt-get install -y --no-install-recommends \
   netcat-traditional
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir -U \
   setuptools \
   wheel

# Clone wyoming-piper layer
git clone https://github.com/rhasspy/piper /tmp/wyoming-piper
cd /tmp/wyoming-piper
cmake -Bbuild -DCMAKE_INSTALL_PREFIX=install
cmake --build build --config Release
cmake --install build
# Do a test run
./build/piper --help
mkdir -p piper
cp -dR /tmp/wyoming-piper/build/install/* /tmp/wyoming-piper/piper/
tar -czf "piper_aarch64.tar.gz" /tmp/wyoming-piper/piper/

# pip3 install --no-cache-dir --verbose -r /tmp/wyoming-piper/requirements.txt
# pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/wyoming-piper
# pip3 install --no-cache-dir --verbose /opt/wyoming_piper*.whl
# pip3 show wyoming_piper
# rm -rf /tmp/wyoming-piper

# Clone rootfs
git clone --depth=1 https://github.com/home-assistant/addons /tmp/addons
git -C /tmp/addons sparse-checkout set --no-cone piper/ '!*/piper'

# Fix: Adjust reading config data from config.yaml file when not using HA Supervisor
sed -i \
   -e "s/bashio::config.true 'update_voices'/[[ \"$PIPER_UPDATE_VOICES\" == \"true\" ]]/g" \
   -e "s/bashio::config.true 'debug_logging'/[[ \"$PIPER_DEBUG\" == \"true\" ]]/g" \
   -e "s/--length-scale.*/--length-scale \"$PIPER_LENGTH_SCALE\" \\\\/g" \
   -e "s/--noise-scale.*/--noise-scale \"$PIPER_NOISE_SCALE\" \\\\/g" \
   -e "s/--noise-w.*/--noise-w \"$PIPER_NOISE_W\" \\\\/g" \
   -e "s/--speaker.*/--speaker \"$PIPER_SPEAKER\" \\\\/g" \
   -e "s/--voice.*/--voice \"$PIPER_VOICE\" \\\\/g" \
   -e "s/--max-piper-procs.*/--max-piper-procs \"$PIPER_MAX_PROC\" \\\\/g" \
   /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/piper/run

# Enable CUDA
sed -i '15a\--cuda \' /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/piper/run
cat /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/piper/run

# Fix: Disable native Discovery handled by HA Supervisor
sed -i '$d' /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
sed -i '$d' /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run
cat /tmp/addons/piper/rootfs/etc/s6-overlay/s6-rc.d/discovery/run

cp -r /tmp/addons/piper/rootfs/* /
cp /tmp/addons/piper/config.yaml /etc/s6-overlay/s6-rc.d/piper/config.yaml
rm -rf /tmp/addons

python3 -c 'import wyoming_piper; print(wyoming_piper.__version__);'
