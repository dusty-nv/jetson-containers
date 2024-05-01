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

# Do a piper-tts test run
/opt/piper/build/piper --help

# Clone wyoming-piper layer
git clone --branch=${WYOMING_PIPER_VERSION} https://github.com/rhasspy/wyoming-piper /tmp/wyoming-piper
# Enable CUDA
git -C /tmp/wyoming-piper apply /tmp/wyoming/wyoming-piper_cuda_path.diff
git -C /tmp/wyoming-piper status
pip3 install --no-cache-dir --verbose -r /tmp/wyoming-piper/requirements.txt
pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/wyoming-piper
pip3 install --no-cache-dir --verbose /opt/wyoming_piper*.whl
pip3 show wyoming_piper
rm -rf /tmp/wyoming-piper

python3 -c 'import wyoming_piper; print(wyoming_piper.__version__);'
