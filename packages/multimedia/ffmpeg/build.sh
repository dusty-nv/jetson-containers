#!/usr/bin/env bash
set -ex
cd /opt

# Where dependencies will install
PREFIX="/usr/local"

# Source + a versioned, absolute dist dir for FFmpeg
SOURCE="/opt/ffmpeg"
DIST="/opt/ffmpeg/dist"
# pkg-config search path (include both /usr/local and our dist)
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${DIST}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "BUILDING FFMPEG $FFMPEG_VERSION to $DIST"
wget $WGET_FLAGS https://www.ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.gz
tar -xvzf ffmpeg-$FFMPEG_VERSION.tar.gz

mv ffmpeg-${FFMPEG_VERSION} ffmpeg
cd ffmpeg

# deps...
apt-get update && apt-get install -y --no-install-recommends \
  autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev \
  libgnutls28-dev libmp3lame-dev libsdl2-dev libtool libva-dev libvdpau-dev \
  libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev libvpx-dev \
  libx264-dev libx265-dev libopus-dev meson ninja-build pkg-config \
  texinfo wget yasm nasm zlib1g-dev libc6 libc6-dev unzip libnuma1 libnuma-dev \
  libunistring-dev nettle-dev libgmp-dev libidn2-0-dev && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# libaom for AV1
git clone https://aomedia.googlesource.com/aom
mkdir aom/builder
cd aom/builder

cmake -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX="$DIST" \
  -DENABLE_TESTS=OFF \
  -DENABLE_NASM=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DAOM_EXTRA_C_FLAGS="-fno-lto" \
  ../

make -j$(nproc)
make install

# libsvtav1 for AV1
# temporal version 2.3.0, they breaks ffmpeg build: https://gitlab.com/AOMediaCodec/SVT-AV1/-/commit/988e930c1083ce518ead1d364e3a486e9209bf73#900962ec0dfb11881a5f25ce6fcad8e815c8fd45_1056_1122
# solution mid-february: https://gitlab.com/AOMediaCodec/SVT-AV1/-/merge_requests/2355#note_2312506245
# solved https://gitlab.com/AOMediaCodec/SVT-AV1/-/merge_requests/2387
cd $SOURCE

git -C SVT-AV1 pull 2> /dev/null || \
git clone --recursive https://gitlab.com/AOMediaCodec/SVT-AV1.git -b v2.3.0

mkdir SVT-AV1/build
cd SVT-AV1/build

cmake -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX="$DIST" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  ../

make -j$(nproc)
make install

# dav1d for AV1 decode
# FFmpeg 7+/8+ requires dav1d >= 1.0.0, but Ubuntu 22.04 (JetPack 6 / L4T r36)
# only ships libdav1d 0.9.2. Build a modern dav1d into $DIST so configure can
# find it via PKG_CONFIG_PATH. Do not install the system libdav1d-dev package
# for this path — it is too old and can confuse version checks.
cd $SOURCE
DAV1D_VERSION="${DAV1D_VERSION:-1.5.1}"
git clone --depth 1 -b "${DAV1D_VERSION}" https://github.com/videolan/dav1d.git
mkdir dav1d/build
cd dav1d/build

# --libdir=lib keeps pkg-config files under $DIST/lib/pkgconfig on multiarch.
meson setup .. \
  --prefix="$DIST" \
  --libdir=lib \
  --buildtype=release \
  --default-library=shared \
  -Denable_tools=false \
  -Denable_tests=false

ninja -j"$(nproc)"
ninja install

# make these discoverable to ffmpeg build
export PKG_CONFIG_PATH="$DIST/lib/pkgconfig:$PKG_CONFIG_PATH"

pkg-config --modversion aom
pkg-config --modversion SvtAv1Enc
pkg-config --modversion dav1d

# Use the Video Codec SDK headers already installed by the video-codec-sdk
# dependency (pinned, e.g. n13.0.19.0). Do NOT re-clone nv-codec-headers from
# master: newer header revisions renamed NV_ENC_CLOCK_TIMESTAMP_SET fields
# (countingType -> countingTypeLSB), which breaks FFmpeg 8.1's nvenc.c.
if ! pkg-config --exists ffnvcodec; then
  echo "ERROR: ffnvcodec.pc not found. The video-codec-sdk package must be installed first."
  exit 1
fi
echo "Using ffnvcodec $(pkg-config --modversion ffnvcodec)"

export PATH=/usr/local/cuda/bin:${PATH}

# Only enable GPU architectures supported by the installed nvcc.
# A static list that includes sm_88 / sm_100+ (added for CUDA 13 / newer dGPUs)
# causes `nvcc fatal: Unsupported gpu architecture` on Jetson Orin with CUDA 12.6.
NVCCFLAGS="-std=c++17 -O3"
if command -v nvcc >/dev/null 2>&1; then
  SUPPORTED_CODES="$(nvcc --list-gpu-code 2>/dev/null || true)"
  for sm in 75 80 86 87 88 89 90 100 103 110 120 121; do
    if echo "${SUPPORTED_CODES}" | grep -q "sm_${sm}"; then
      NVCCFLAGS="${NVCCFLAGS} -gencode arch=compute_${sm},code=sm_${sm}"
    fi
  done
fi
echo "Using NVCCFLAGS: ${NVCCFLAGS}"

# Build FFmpeg
cd $SOURCE

./configure \
  --prefix="$DIST" \
  --extra-cflags="-I${DIST}/include -I/usr/local/cuda/include -O3 -fPIC" \
  --extra-cxxflags="-std=c++17" \
  --extra-ldflags="-L${DIST}/lib -fno-lto -L/usr/local/cuda/lib64" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="${DIST}/bin" \
  --disable-doc \
  --disable-static \
  --enable-shared \
  --enable-gnutls \
  --enable-libvpx \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libmp3lame \
  --enable-libfreetype \
  --enable-libass \
  --enable-libaom \
  --enable-libsvtav1 \
  --enable-libdav1d \
  --extra-cflags=-I/usr/local/cuda/include \
  --extra-ldflags=-L/usr/local/cuda/lib64 \
  --enable-nvenc \
  --enable-nvdec \
  --enable-cuda \
  --enable-cuvid \
  --nvccflags="$NVCCFLAGS"

make -j"$(nproc)"
make install

DIST_ABS="$(realpath "$DIST")"
echo "FFmpeg built and installed to $DIST_ABS"
test -x "${DIST_ABS}/bin/ffmpeg" || { echo "FFmpeg binary not found in ${DIST_ABS}/bin"; exit 1; }
tarpack upload "ffmpeg-${FFMPEG_VERSION}" "${DIST_ABS}" || echo "failed to upload tarball"

# Optionally install into /usr/local for runtime
cp -r "${DIST_ABS}/"* /usr/local/

# Fix pkg-config files to use /usr/local instead of /opt/ffmpeg/dist
if [ -d /usr/local/lib/pkgconfig ]; then
  sed -i "s|${DIST_ABS}|/usr/local|g" /usr/local/lib/pkgconfig/*.pc 2>/dev/null || true
fi

ldconfig
