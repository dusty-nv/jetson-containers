#!/usr/bin/env bash
# wyoming-piper
set -ex

pip3 install -U \
   build \
   wheel

git clone --branch=${WYOMING_PIPER_BRANCH} https://github.com/rhasspy/wyoming-piper /tmp/wyoming-piper
cd /tmp/wyoming-piper

git apply /tmp/wyoming/piper/wyoming-piper-cuda.diff
git status

python -m build --wheel --outdir $PIP_WHEEL_DIR

cd /
rm -rf /tmp/wyoming-piper

pip3 install $PIP_WHEEL_DIR/wyoming_piper*.whl

pip3 show wyoming_piper
python3 -c 'import wyoming_piper; print(wyoming_piper.__version__);'

twine upload --verbose $PIP_WHEEL_DIR/wyoming_piper*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/wyoming_piper*.whl