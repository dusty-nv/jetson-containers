#!/usr/bin/env bash
set -ex
echo "Building torchcodec ${TORCHCODEC_VERSION}"

# --- Install required system dependencies ---
apt-get update
apt-get install -y --no-install-recommends \
    git \
    pkg-config \
    libffi-dev \
    libsndfile1 \
    python${PYTHON_VERSION}-dev

rm -rf /var/lib/apt/lists/*
apt-get clean

# --- Clone torchcodec repository (try versioned tags first, fallback to release branch, then master) ---
git clone --branch=v${TORCHCODEC_VERSION} --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec \
  || git clone --branch=release/${BRANCH_VERSION} --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec \
  || git clone --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec

cd /opt/torchcodec

export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
export ENABLE_CUDA=1

# --- Build wheel ---
# sed -i 's/-Werror//g' /opt/torchcodec/src/torchcodec/_core/CMakeLists.txt

# (opcional pero útil) bajar el nivel del warning
export CXXFLAGS="$CXXFLAGS -Wno-deprecated-declarations"
export CFLAGS="$CFLAGS -Wno-deprecated-declarations"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib/$(uname -m)-linux-gnu/pkgconfig:/usr/lib/$(uname -m)-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"

pkg-config --variable pc_path pkg-config | tr ':' '\n'
pkg-config --debug libavcodec |& sed -n '1,160p' | grep -E "Searching|Looking|Trying|Located|open" || true

# If there are .pc in /usr/local, rewrite them to prefix=/usr/local
for pc in libavcodec libavformat libavutil libswresample libswscale libavdevice libavfilter; do
  f="/usr/local/lib/pkgconfig/${pc}.pc"
  if [ -f "$f" ]; then
    sed -i 's|^prefix=.*|prefix=/usr/local|' "$f"
    # normalize includedir/libdir if they were absolute
    sed -i 's|^includedir=.*|includedir=${prefix}/include|' "$f" || true
    sed -i 's|^libdir=.*|libdir=${prefix}/lib|' "$f" || true
  fi
done

if [ ! -d /opt/ffmpeg/dist ] && [ -d /usr/local/include ] && [ -d /usr/local/lib ]; then
  mkdir -p /opt/ffmpeg
  ln -sfn /usr/local /opt/ffmpeg/dist
fi

BUILD_VERSION=${TORCHCODEC_VERSION} \
BUILD_SOX=1 \
python3 setup.py bdist_wheel --verbose --dist-dir /opt

cd ../
rm -rf /opt/torchcodec

# --- Install and verify ---
uv pip install /opt/torchcodec*.whl
uv pip show torchcodec && python3 -c 'import torchcodec; print(torchcodec.__version__);'

# --- Upload (if configured) ---
twine upload --verbose /opt/torchcodec*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
