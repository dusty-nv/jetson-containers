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
git clone --branch=v${WYOMING_WHISPER_VERSION} https://github.com/rhasspy/wyoming-faster-whisper /tmp/wyoming-faster-whisper
pip3 install --no-cache-dir --verbose -r /tmp/wyoming-faster-whisper/requirements.txt
pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/wyoming-faster-whisper
pip3 install --no-cache-dir --verbose /opt/wyoming_faster_whisper*.whl
pip3 show wyoming_faster_whisper
rm -rf /tmp/wyoming-faster-whisper

python3 -c 'import wyoming_faster_whisper; print(wyoming_faster_whisper.__version__);'
