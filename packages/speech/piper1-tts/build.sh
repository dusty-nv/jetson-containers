#!/usr/bin/env bash
set -exu

echo "Building piper1-tts ${PIPER_VERSION} (${PIPER_BRANCH})"

apt-get update
apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    ninja-build
rm -rf /var/lib/apt/lists/*
apt-get clean

git clone --branch ${PIPER_BRANCH} --depth 1 https://github.com/OHF-voice/piper1-gpl /opt/piper1-tts
cd /opt/piper1-tts

sed -i \
    -e 's|"onnxruntime.*",||g' \
    setup.py

# Fix the package version
codebase_version=$(grep -Po '(?<=version=")[^"]+' setup.py)
if [ "$codebase_version" != "${PIPER_VERSION}" ]; then
    echo "Fixed piper1-tts codebase version: ${codebase_version} --> ${PIPER_VERSION}"
    sed -i "s/\(version=\"\)[^\"]+/\1${PIPER_VERSION}/" setup.py
fi

uv pip install --no-cache-dir --verbose build==1.2.2 scikit-build

python3 setup.py build_ext --inplace
python3 -m build --sdist --wheel --outdir ${PIP_WHEEL_DIR}

uv pip install --no-cache-dir --verbose ${PIP_WHEEL_DIR}/piper_tts*.whl
uv pip show piper-tts

# upload wheels
twine upload --verbose ${PIP_WHEEL_DIR}/piper_tts*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
