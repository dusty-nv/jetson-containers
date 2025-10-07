#!/usr/bin/env bash
# wyoming-openwakeword
set -ex

apt-get update
apt-get install -y --no-install-recommends \
   netcat-traditional \
   libopenblas0
apt-get clean
rm -rf /var/lib/apt/lists/*

uv pip install --upgrade \
   setuptools \
   wheel

echo "wyoming-openwakeword: ${WYOMING_OPENWAKEWORD_VERSION} (branch: ${WYOMING_OPENWAKEWORD_BRANCH})"

git clone --branch=${WYOMING_OPENWAKEWORD_BRANCH} https://github.com/rhasspy/wyoming-openwakeword /opt/wyoming-openwakeword
cd /opt/wyoming-openwakeword

python3 setup.py sdist bdist_wheel --verbose --dist-dir $PIP_WHEEL_DIR

cd /
rm -rf /opt/wyoming-openwakeword

uv pip install $PIP_WHEEL_DIR/wyoming_openwakeword*.whl

uv pip show wyoming_openwakeword
python3 -c 'import wyoming_openwakeword; print(wyoming_openwakeword.__version__);'

twine upload --verbose $PIP_WHEEL_DIR/wyoming_openwakeword*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/wyoming_openwakeword*.whl
