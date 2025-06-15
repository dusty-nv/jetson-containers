#!/usr/bin/env bash
set -ex
cd /opt

SOURCE="/opt/ffmpeg"
DIST="$SOURCE/dist"

echo "BUILDING FFMPEG $FFMPEG_VERSION to $DIST"
wget $WGET_FLAGS https://www.ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.gz
tar -xvzf ffmpeg-$FFMPEG_VERSION.tar.gz

mv ffmpeg-${FFMPEG_VERSION} ffmpeg
cd ffmpeg

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

# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies
apt-get update
apt-get install -y --no-install-recommends \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  libmp3lame-dev \
  libsdl2-dev \
  libtool \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  libvpx-dev \
  libx264-dev \
  libx265-dev \
  libopus-dev \
  libdav1d-dev \
  meson \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  nasm \
  zlib1g-dev
apt-get clean
rm -rf /var/lib/apt/lists/*

# Build FFmpeg
cd $SOURCE

./configure \
  --prefix="$DIST" \
  --extra-cflags="-I$DIST/include -fno-lto" \
  --extra-ldflags="-L$DIST/lib -fno-lto" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$DIST/bin" \
  --disable-doc \
  --enable-shared \
  --enable-gpl \
  --enable-gnutls \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libvpx \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libmp3lame \
  --enable-libfreetype \
  --enable-libass \
  --enable-libaom \
  --enable-libsvtav1 \
  --enable-libdav1d

make -j$(nproc)
make install

# upload to jetson-ai-lab build cache
tarpack upload ffmpeg-${FFMPEG_VERSION} $DIST/ || echo "failed to upload tarball"

# install it like cached builds
cp -r $DIST/* /usr/local/
