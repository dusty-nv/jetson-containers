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
export PKG_CONFIG_PATH="/usr/lib/${aarch}-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig:${DIST}/lib/pkgconfig:/usr/lib/pkgconfig"

export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
export ENABLE_CUDA=1

# --- Build wheel ---
# sed -i 's/-Werror//g' /opt/torchcodec/src/torchcodec/_core/CMakeLists.txt

# (opcional pero útil) bajar el nivel del warning
export CXXFLAGS="$CXXFLAGS -Wno-deprecated-declarations"
export CFLAGS="$CFLAGS -Wno-deprecated-declarations"

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
