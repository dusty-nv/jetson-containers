#!/usr/bin/env bash
# wyoming-whisper
set -ex

apt-get update
apt-get install -y --no-install-recommends \
   netcat
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir -U \
   setuptools \
   wheel

# Clone wyoming-faster-whisper layer
git clone --branch=${WYOMING_WHISPER_BRANCH} https://github.com/rhasspy/wyoming-faster-whisper /tmp/wyoming-faster-whisper

cd /tmp/wyoming-faster-whisper

sed -i \
   -e 's|^faster-whisper.*||g' \
   requirements.txt
cat requirements.txt

python3 setup.py sdist bdist_wheel --verbose --dist-dir /opt/wheels

cd /
rm -rf /tmp/wyoming-faster-whisper

pip3 install --no-cache-dir --verbose /opt/wheels/wyoming_faster_whisper*.whl

pip3 show wyoming_faster_whisper
python3 -c 'import wyoming_faster_whisper; print(wyoming_faster_whisper.__version__);'

twine upload --skip-existing --verbose /opt/wheels/wyoming_faster_whisper*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm /opt/wheels/wyoming_faster_whisper*.whl