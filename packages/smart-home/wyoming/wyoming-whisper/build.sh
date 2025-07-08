#!/usr/bin/env bash
# wyoming-whisper
set -ex

pip3 install -U \
   build \
   wheel

# Clone wyoming-faster-whisper layer
git clone --branch=${WYOMING_WHISPER_BRANCH} https://github.com/rhasspy/wyoming-faster-whisper /tmp/wyoming-faster-whisper
cd /tmp/wyoming-faster-whisper

sed -i \
   -e 's|"faster-whisper.*"||g' \
   pyproject.toml

python -m build --wheel --outdir $PIP_WHEEL_DIR

cd /
rm -rf /tmp/wyoming-faster-whisper

pip3 install $PIP_WHEEL_DIR/wyoming_faster_whisper*.whl

pip3 show wyoming_faster_whisper
python3 -c 'import wyoming_faster_whisper; print(wyoming_faster_whisper.__version__);'

twine upload --verbose $PIP_WHEEL_DIR/wyoming_faster_whisper*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/wyoming_faster_whisper*.whl