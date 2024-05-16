#!/usr/bin/env bash
# homeassistant-supervisor

set -xo pipefail

curl -Lso /usr/bin/yq "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_arm64"
curl -Lso /usr/bin/cosign "https://github.com/home-assistant/cosign/releases/download/${COSIGN_VERSION}/cosign_${BUILD_ARCH}"
chmod a+x /usr/bin/cosign

# Install Home Assistant Supervisor
echo "Installing Home Assistant Supervisor ${SUPERVISOR_VERSION}..."
git clone --branch=${SUPERVISOR_VERSION} https://github.com/home-assistant/supervisor /usr/src/supervisor

# Set version
sed -i \
  -e "s|99.9.9dev|${SUPERVISOR_VERSION}|g" \
  /usr/src/supervisor/supervisor/const.py
sed -i \
  -e "s|version=.*|version='${SUPERVISOR_VERSION}',|g" \
  /usr/src/supervisor/setup.py

cat /usr/src/supervisor/supervisor/const.py | grep "SUPERVISOR_VERSION ="
cat /usr/src/supervisor/setup.py | grep "version="

export MAKEFLAGS="-j$(nproc)" 
pip3 install --no-cache-dir --verbose -r /usr/src/supervisor/requirements.txt

pip3 install -e /usr/src/supervisor
python3 -m compileall /usr/src/supervisor/supervisor

pip3 show supervisor
python3 -c 'import supervisor'

# Copy Home Assistant Supervisor S6-Overlay
cp -r /usr/src/supervisor/rootfs/* /
