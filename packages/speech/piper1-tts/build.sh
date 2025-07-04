#!/usr/bin/env bash
set -ex

echo "Building piper1-tts ${PIPER_VERSION} (${PIPER_BRANCH})"

apt-get update
apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    ninja-build \
    swig
rm -rf /var/lib/apt/lists/*
apt-get clean

git clone --branch ${PIPER_BRANCH} --depth 1 https://github.com/OHF-voice/piper1-gpl /opt/piper1-tts
cd /opt/piper1-tts

sed -i \
    -e 's|"onnxruntime.*",||g' \
    setup.py

pip3 install --no-cache-dir --verbose build==1.2.2 scikit-build flask

# python3 setup.py build_ext --inplace
python3 -m build --wheel --outdir ${PIP_WHEEL_DIR}

pip3 install --no-cache-dir --verbose ${PIP_WHEEL_DIR}/piper*.whl
pip3 show piper

# upload wheels
twine upload --verbose ${PIP_WHEEL_DIR}/piper*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
