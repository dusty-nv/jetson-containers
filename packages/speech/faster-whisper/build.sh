#!/usr/bin/env bash
# faster-whisper

set -ex

echo "Building faster-whisper ${FASTER_WHISPER_VERSION} (branch=${FASTER_WHISPER_BRANCH})"

git clone --branch=${FASTER_WHISPER_BRANCH} https://github.com/guillaumekln/faster-whisper /opt/faster-whisper
cd /opt/faster-whisper

sed -i \
   -e 's|^onnxruntime.*||g' \
   -e 's|^huggingface_hub.*||g' \
   -e 's|^ctranslate2.*||g' \
   requirements.txt

echo "__version__ = \"${FASTER_WHISPER_VERSION}\"" > faster_whisper/version.py

uv pip install -U -r requirements.txt

python3 setup.py --verbose bdist_wheel --dist-dir $PIP_WHEEL_DIR
uv pip install $PIP_WHEEL_DIR/faster_whisper*.whl

uv pip show faster_whisper
python3 -c 'import faster_whisper; print(faster_whisper.__version__);'

twine upload --verbose $PIP_WHEEL_DIR/faster_whisper*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/faster_whisper*.whl
