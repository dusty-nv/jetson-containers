#!/usr/bin/env bash
# homeassistant-core
set -ex

# Install uv & ruff
pip3 install --no-cache-dir --verbose uv==0.1.27 ruff

# Install psutil-home-assistant
git clone https://github.com/home-assistant-libs/psutil-home-assistant /tmp/psutil-home-assistant

pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/psutil-home-assistant
pip3 install --no-cache-dir --verbose /opt/psutil_home_assistant*.whl
pip3 show psutil-home-assistant

python3 -c 'import psutil_home_assistant;'
rm -rf /tmp/psutil-home-assistant

# Install homeassistant-core
echo "HA_BRANCH: ${HA_BRANCH}"
pip3 install --no-cache-dir --ignore-installed blinker

git clone --branch=${HA_BRANCH} https://github.com/home-assistant/core homeassistant
uv pip install --no-cache-dir --verbose -r homeassistant/requirements.txt

ln -s /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 /usr/local/lib/libjemalloc.so.2

if ls homeassistant/home_assistant_*.whl 1> /dev/null 2>&1; then
	uv pip install --no-cache-dir --verbose homeassistant/home_assistant_*.whl;
fi

LD_PRELOAD="${LD_PRELOAD}:/usr/local/lib/libjemalloc.so.2" \
MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000" \
uv pip install --no-cache-dir --verbose -r homeassistant/requirements_all.txt

uv pip install -e homeassistant
python3 -m compileall homeassistant

# Generate languages
cd homeassistant

python3 -m script.hassfest
python3 -m script.translations develop --all

# Home Assistant S6-Overlay
cp -r rootfs/* /