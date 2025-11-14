#!/usr/bin/env bash
# wyoming-piper
set -ex

echo "Building wyoming-piper ${WYOMING_PIPER_VERSION} (${WYOMING_PIPER_BRANCH})"

uv pip install -U \
   build \
   wheel \
   zeroconf

git clone --branch=${WYOMING_PIPER_BRANCH} https://github.com/rhasspy/wyoming-piper /tmp/wyoming-piper
cd /tmp/wyoming-piper

sed -i.bak -E \
  -e 's/"piper-tts==[^"]+"/"piper-tts"/' \
  pyproject.toml

python3 -m build --wheel --outdir $PIP_WHEEL_DIR

cd /
rm -rf /tmp/wyoming-piper

uv pip install $PIP_WHEEL_DIR/wyoming_piper*.whl

uv pip show wyoming_piper
python3 -c 'import wyoming_piper; print(wyoming_piper.__version__);'

twine upload --verbose $PIP_WHEEL_DIR/wyoming_piper*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/wyoming_piper*.whl
