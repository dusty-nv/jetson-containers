#!/usr/bin/env bash
set -ex
echo "Building torchcodec ${TORCHCODEC_VERSION}"

# --- Install required system dependencies ---
apt-get update
apt-get install -y --no-install-recommends \
    git \
    pkg-config \
    libffi-dev \
    libsndfile1

rm -rf /var/lib/apt/lists/*
apt-get clean

# --- Clone torchcodec repository (try versioned tags first, fallback to release branch, then master) ---
git clone --branch=v${TORCHCODEC_VERSION} --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec \
  || git clone --branch=release/${BRANCH_VERSION} --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec \
  || git clone --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec

cd /opt/torchcodec
# --- Build wheel ---
BUILD_VERSION=${TORCHCODEC_VERSION} \
BUILD_SOX=1 \
python3 setup.py bdist_wheel --verbose --dist-dir /opt

cd ../
rm -rf /opt/torchcodec

# --- Install and verify ---
pip3 install /opt/torchcodec*.whl
pip3 show torchcodec && python3 -c 'import torchcodec; print(torchcodec.__version__);'

# --- Upload (if configured) ---
twine upload --verbose /opt/torchcodec*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
