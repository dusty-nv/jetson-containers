#!/usr/bin/env bash
set -ex
cd /opt

# No custom DIST, install directly to /usr/local
PREFIX="/usr/local"

# Update pkg-config path if needed (usually /usr/local is included, but to be safe)
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "BUILDING FFMPEG $FFMPEG_VERSION to $PREFIX"
wget $WGET_FLAGS https://www.ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.gz
tar -xvzf ffmpeg-$FFMPEG_VERSION.tar.gz

mv ffmpeg-${FFMPEG_VERSION} ffmpeg
cd ffmpeg

# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies
apt-get update
apt-get update && apt-get install -y --no-install-recommends \
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
  zlib1g-dev \
  libc6 \
  libc6-dev \
  unzip \
  libnuma1 \
  libnuma-dev \
  pkg-config \
  libunistring-dev \
  libgnutls28-dev \
  nettle-dev \
  libgmp-dev \
  libidn2-0-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# libaom for AV1
git clone https://aomedia.googlesource.com/aom
mkdir aom/builder
cd aom/builder

cmake -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DENABLE_TESTS=OFF \
  -DENABLE_NASM=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DAOM_EXTRA_C_FLAGS="-fno-lto" \
  ../

make -j$(nproc)
make install

# libsvtav1 for AV1
cd /opt/ffmpeg

git -C SVT-AV1 pull 2> /dev/null || \
git clone --recursive https://gitlab.com/AOMediaCodec/SVT-AV1.git -b v2.3.0

mkdir SVT-AV1/build
cd SVT-AV1/build

cmake -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  ../

make -j$(nproc)
make install

pkg-config --modversion aom
pkg-config --modversion SvtAv1Enc

git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers && make PREFIX="$PREFIX" install

# Build FFmpeg
cd /opt/ffmpeg

export PATH=/usr/local/cuda/bin:${PATH}
nvcc --version
gcc --version
# For DGX Spark (Blackwell, compute capability 12.1)
# For RTX 5000 Series (Blackwell, compute capability 12.0)
# For Jetson Thor (Blackwell, compute capability 11.0)
# For RTX GB300 Series (Blackwell, compute capability 10.3)
# For RTX GB200 Series (Blackwell, compute capability 10.0)
# For H100 Series (Hopper, compute capability 9.0)
# For RTX 4000 Series (Ada Lovelace, compute capability 8.9)
# For RTX 5000, 3000 Series (Ampere, compute capability 8.6)
# For RTX 3000 Series (Ampere, compute capability 8.0)
# For RTX 2000 Series (Turing, compute capability 7.5)

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

# NO COMPLIANCE: https://www.ffmpeg.org/legal.html
# ./configure \
#   --prefix="$DIST" \
#   --extra-cflags="-I$DIST/include -I/usr/local/cuda/include -O3 -fPIC" \
#   --extra-cxxflags="-std=c++17" \
#   --extra-ldflags="-L$DIST/lib -fno-lto -L/usr/local/cuda/lib64" \
#   --extra-libs="-lpthread -lm" \
#   --ld="g++" \
#   --bindir="$DIST/bin" \
#   --disable-doc \
#   --disable-static \
#   --enable-shared \
#   --enable-gpl \
#   --enable-nonfree \
#   --enable-gnutls \
#   --enable-libx264 \
#   --enable-libx265 \
#   --enable-libvpx \
#   --enable-libopus \
#   --enable-libvorbis \
#   --enable-libmp3lame \
#   --enable-libfreetype \
#   --enable-libass \
#   --enable-libaom \
#   --enable-libsvtav1 \
#   --enable-libdav1d \
#   --enable-nvenc \
#   --enable-nvdec \
#   --enable-cuda \
#   --nvccflags="$NVCCFLAGS"

./configure \
  --prefix="$DIST" \
  --extra-cflags="-I$DIST/include -I/usr/local/cuda/include -O3 -fPIC" \
  --extra-cxxflags="-std=c++17" \
  --extra-ldflags="-L$DIST/lib -fno-lto -L/usr/local/cuda/lib64" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$DIST/bin" \
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
  --enable-nvenc \
  --enable-nvdec \
  --enable-cuda \
  --enable-cuvid \
  --nvccflags="$NVCCFLAGS"

make -j$(nproc)
make install

# Update library cache
ldconfig

# upload to jetson-ai-lab build cache
tarpack upload ffmpeg-${FFMPEG_VERSION} $DIST/ || echo "failed to upload tarball"

# install it like cached builds
cp -r $DIST/* /usr/local/
