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

# --- Clone torchcodec repository (try versioned tags first, then RC, release branch, then master) ---
git clone --branch=v${TORCHCODEC_VERSION} --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec \
  || git clone --branch=v${TORCHCODEC_VERSION}-rc1 --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec \
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

# --- Locate FFmpeg .pc files and set PKG_CONFIG_PATH ---
FFMPEG_PC_DIR=""
for search_dir in \
    /usr/local/lib/pkgconfig \
    /usr/local/lib/aarch64-linux-gnu/pkgconfig \
    /opt/ffmpeg/dist/lib/pkgconfig \
    /usr/lib/aarch64-linux-gnu/pkgconfig \
    /usr/lib/pkgconfig; do
  if [ -f "$search_dir/libavcodec.pc" ]; then
    FFMPEG_PC_DIR="$search_dir"
    echo "Found FFmpeg .pc files in $search_dir"
    break
  fi
done

if [ -z "$FFMPEG_PC_DIR" ]; then
  echo "Searching filesystem for libavcodec.pc..."
  FFMPEG_PC_DIR="$(find / -name 'libavcodec.pc' -print -quit 2>/dev/null | xargs -r dirname)"
  [ -n "$FFMPEG_PC_DIR" ] && echo "Found FFmpeg .pc files via search: $FFMPEG_PC_DIR"
fi

if [ -n "$FFMPEG_PC_DIR" ]; then
  export PKG_CONFIG_PATH="${FFMPEG_PC_DIR}:${PKG_CONFIG_PATH:-}"
  for pc in libavcodec libavformat libavutil libswresample libswscale libavdevice libavfilter; do
    f="${FFMPEG_PC_DIR}/${pc}.pc"
    if [ -f "$f" ]; then
      sed -i 's|^prefix=.*|prefix=/usr/local|' "$f"
      sed -i 's|^includedir=.*|includedir=${prefix}/include|' "$f" || true
      sed -i 's|^libdir=.*|libdir=${prefix}/lib|' "$f" || true
    fi
  done
else
  echo "WARNING: Could not locate FFmpeg .pc files anywhere"
fi

echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
pkg-config --variable pc_path pkg-config | tr ':' '\n'
pkg-config --debug libavcodec |& sed -n '1,160p' | grep -E "Searching|Looking|Trying|Located|open" || true

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
