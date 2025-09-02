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
  libx264-dev libx265-dev libopus-dev libdav1d-dev meson ninja-build pkg-config \
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

# make these discoverable to ffmpeg build
export PKG_CONFIG_PATH="$DIST/lib/pkgconfig:$PKG_CONFIG_PATH"

pkg-config --modversion aom
pkg-config --modversion SvtAv1Enc

# nv-codec-headers
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers && make PREFIX="$DIST" install

export PATH=/usr/local/cuda/bin:${PATH}
NVCCFLAGS="\
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_87,code=sm_87 \
-gencode arch=compute_88,code=sm_88 \
-gencode arch=compute_89,code=sm_89 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_100,code=sm_100 \
-gencode arch=compute_103,code=sm_103 \
-gencode arch=compute_110,code=sm_110 \
-gencode arch=compute_120,code=sm_120 \
-gencode arch=compute_121,code=sm_121 \
-std=c++17 -O3"

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
ldconfig
