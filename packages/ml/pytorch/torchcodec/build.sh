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
ARCH="$(uname -m)"

if [ ! -d /opt/ffmpeg/dist ] && [ -d /usr/local/include ] && [ -d /usr/local/lib ]; then
  mkdir -p /opt/ffmpeg
  ln -sfn /usr/local /opt/ffmpeg/dist
fi

if [ -f /usr/local/lib/pkgconfig/libavcodec.pc ]; then
  for pc in libavcodec libavformat libavutil libswresample libswscale libavdevice libavfilter; do
    f="/usr/local/lib/pkgconfig/${pc}.pc"
    if [ -f "$f" ]; then
      sed -i 's|^prefix=.*|prefix=/usr/local|' "$f"
      sed -i 's|^includedir=.*|includedir=${prefix}/include|' "$f" || true
      sed -i 's|^libdir=.*|libdir=${prefix}/lib|' "$f" || true
    fi
  done
fi

PKG_PATHS=(
  "/usr/local/lib/pkgconfig"
  "/usr/local/lib64/pkgconfig"
  "/usr/local/lib/${ARCH}-linux-gnu/pkgconfig"
  "/usr/lib/${ARCH}-linux-gnu/pkgconfig"
  "/usr/lib/pkgconfig"
  "/opt/ffmpeg/dist/lib/pkgconfig"
  "/opt/ffmpeg/lib/pkgconfig"
)

PKG_PATH_JOINED=""
for d in "${PKG_PATHS[@]}"; do
  if [ -d "$d" ]; then
    if [ -z "$PKG_PATH_JOINED" ]; then
      PKG_PATH_JOINED="$d"
    else
      PKG_PATH_JOINED="${PKG_PATH_JOINED}:$d"
    fi
  fi
done

if [ -n "$PKG_PATH_JOINED" ]; then
  export PKG_CONFIG_PATH="${PKG_PATH_JOINED}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
fi

if ! pkg-config --exists libavdevice libavfilter libavformat libavcodec libavutil libswresample libswscale; then
  apt-get update
  apt-get install -y --no-install-recommends \
      libavcodec-dev \
      libavdevice-dev \
      libavfilter-dev \
      libavformat-dev \
      libavutil-dev \
      libswresample-dev \
      libswscale-dev
  rm -rf /var/lib/apt/lists/*
  apt-get clean
  export PKG_CONFIG_PATH="/usr/lib/${ARCH}-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

pkg-config --modversion libavdevice libavfilter libavformat libavcodec libavutil libswresample libswscale

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
