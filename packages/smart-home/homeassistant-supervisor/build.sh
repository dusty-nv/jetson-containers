#!/usr/bin/env bash
# homeassistant-supervisor

set -euxo pipefail

echo "Installing Home Assistant Supervisor ${SUPERVISOR_VERSION}..."

# Install dependencies
apt-get update
apt-get install -y --no-install-recommends \
	findutils \
	git \
	libffi-dev \
	libpulse-dev \
	musl-tools \
	openssl \
	libyaml-dev
apt-get clean
rm -rf /var/lib/apt/lists/*

curl -Lso /usr/bin/cosign "https://github.com/home-assistant/cosign/releases/download/${COSIGN_VERSION}/cosign_${BUILD_ARCH}"
chmod a+x /usr/bin/cosign

git clone --branch=${SUPERVISOR_VERSION} https://github.com/home-assistant/supervisor /usr/src/supervisor

# Do some fixes
sed -i \
  -e 's|name = "Supervisor"|name = "supervisor"|g' \
  /usr/src/supervisor/pyproject.toml
sed -i \
  -e "s|99.9.9dev|${SUPERVISOR_VERSION}|g" \
  /usr/src/supervisor/supervisor/const.py
sed -i \
  -e "s|version=.*|version='${SUPERVISOR_VERSION}',|g" \
  /usr/src/supervisor/setup.py

export MAKEFLAGS="-j$(nproc)" 
pip3 install --no-cache-dir --verbose -r /usr/src/supervisor/requirements.txt
pip3 install -e /usr/src/supervisor
python3 -m compileall /usr/src/supervisor/supervisor

pip3 show supervisor
python3 -c 'import supervisor'

# Copy Home Assistant Supervisor S6-Overlay
cp -r /usr/src/supervisor/rootfs/* /
