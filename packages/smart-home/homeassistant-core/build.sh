#!/usr/bin/env bash
# homeassistant-core
set -euxo pipefail

echo "Installing Home Assistant Core ${HA_VERSION}..."

apt-get update
apt-get install -y --no-install-recommends \
    autoconf \
    libpcap0.8 \
    libturbojpeg \
    tcl \
    zlib1g-dev \
    ffmpeg
apt-get clean
rm -rf /var/lib/apt/lists/*

python3 -m sqlite3 -v

# Download and build SQLite
mkdir -p /tmp/sqlite
wget $WGET_FLAGS \
    https://github.com/sqlite/sqlite/archive/refs/tags/version-${SQLITE_VERSION}.tar.gz \
    -O /tmp/sqlite/sqlite.tar.gz
# Extract and build SQLite
tar -xzf /tmp/sqlite/sqlite.tar.gz -C /tmp/sqlite
cd /tmp/sqlite/sqlite-version-${SQLITE_VERSION}
./configure --prefix=/usr/local
make -j$(nproc)
make install

ln -sf /usr/local/lib/libsqlite3.so /usr/lib/libsqlite3.so
ln -sf /usr/local/lib/libsqlite3.so.0 /usr/lib/libsqlite3.so.0

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/lib
ldconfig

ls -l /usr/lib/libsqlite3.so
ls -l /usr/lib/libsqlite3.so.0

cd /
rm -rf /tmp/sqlite

python3 -m sqlite3 -v

uv pip install --reinstall blinker
uv pip install uv==0.7.1 ruff

# Install homeassistant-core
git clone --branch=${HA_VERSION} https://github.com/home-assistant/core /usr/src/homeassistant
uv pip install -r /usr/src/homeassistant/requirements_all.txt
uv pip install -e /usr/src/homeassistant
python3 -m compileall /usr/src/homeassistant

# Generate languages
cd /usr/src/homeassistant
python3 -m script.hassfest
python3 -m script.translations develop --all

# Copy Home Assistant S6-Overlay
cp -r rootfs/* /

# Get go2rtc binary
ln -s /data/homeassistant /config
curl -L https://github.com/AlexxIT/go2rtc/releases/download/v1.9.9/go2rtc_linux_arm64 --output /bin/go2rtc
chmod +x /bin/go2rtc
go2rtc --version
