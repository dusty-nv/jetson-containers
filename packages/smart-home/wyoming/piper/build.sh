#!/usr/bin/env bash
# wyoming-piper
set -ex

apt-get update
apt-get install -y --no-install-recommends \
   netcat-traditional
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install -U \
   setuptools \
   wheel

# Do a piper-tts test run
/opt/piper/build/piper --help

# Clone wyoming-piper layer
git clone --branch=${WYOMING_PIPER_BRANCH} https://github.com/rhasspy/wyoming-piper /tmp/wyoming-piper
# Enable CUDA
git -C /tmp/wyoming-piper apply /tmp/wyoming/piper/wyoming-piper_cuda_path.diff
git -C /tmp/wyoming-piper status

pip3 install -r /tmp/wyoming-piper/requirements.txt

# fix version
echo "$WYOMING_PIPER_VERSION" > /tmp/wyoming-piper/wyoming_piper/VERSION
cat /tmp/wyoming-piper/wyoming_piper/VERSION

pip3 wheel --wheel-dir=/opt/wheels --no-deps --verbose /tmp/wyoming-piper
pip3 install /opt/wheels/wyoming_piper*.whl

rm -rf /tmp/wyoming-piper

pip3 show wyoming_piper
python3 -c 'import wyoming_piper; print(wyoming_piper.__version__);'

twine upload --skip-existing --verbose /opt/wheels/wyoming_piper*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm /opt/wheels/wyoming_piper*.whl