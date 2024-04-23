#!/usr/bin/env bash
# wyoming-openwakeword
set -euxo pipefail

apt-get update
apt-get install -y --no-install-recommends \
   netcat-traditional \
   libopenblas0
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir -U \
   setuptools \
   wheel
   
echo "wyoming-openwakeword: ${WYOMING_OPENWAKEWORD_VERSION}"

# TODO: This still uses tflite on CPU / Need to build from scratch on onnxruntime-gpu
pip3 install --no-cache-dir \
   --extra-index-url https://www.piwheels.org/simple \
   "wyoming-openwakeword @ https://github.com/rhasspy/wyoming-openwakeword/archive/refs/tags/v${WYOMING_OPENWAKEWORD_VERSION}.tar.gz"

python3 -c 'import wyoming_openwakeword; print(wyoming_openwakeword.__version__);'
