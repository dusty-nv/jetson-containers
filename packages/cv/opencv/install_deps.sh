#!/usr/bin/env bash
# this script installs build dependencies for compiling OpenCV
set -e -x

ARCH=$(uname -i)
DISTRO=$(lsb_release -rs)

echo "ARCH:   $ARCH"
echo "DISTRO: $DISTRO"

if [[ $DISTRO == "18.04" || $DISTRO == "20.04" ]]; then
	EXTRAS="libavresample-dev libdc1394-22-dev"
fi

if [[ $DISTRO == "24.04" ]]; then
  EXTRAS="libtbbmalloc2 libtbb-dev libopenimageio2.4t64 $EXTRAS"
else
  EXTRAS="libtbb2 libtbb2-dev liblapacke-dev $EXTRAS"
fi

# Ubuntu's FFmpeg headers, unless FFmpeg was already built into the image (ffmpeg package,
# /usr/local). build.sh passes CUDNN_INCLUDE_DIR=/usr/include/<arch>-linux-gnu, which OpenCV adds
# as -isystem, so Ubuntu's older headers there would shadow /usr/local/include: OpenCV then
# compiles against FFmpeg 4.4 and links the newer libraries ('import cv2' fails with undefined
# symbols such as av_stream_get_side_data or avcodec_get_context_defaults3).
FFMPEG_DEV="libavcodec-dev libavformat-dev libswscale-dev libpostproc-dev"
if pkg-config --exists libavcodec && [ "$(pkg-config --variable=includedir libavcodec)" != "/usr/include" ]; then
    echo "using FFmpeg $(pkg-config --modversion libavcodec) from $(pkg-config --variable=includedir libavcodec), not installing $FFMPEG_DEV"
    FFMPEG_DEV=""
fi

apt-get update
apt-get install -y --no-install-recommends \
        libunwind-dev \
        libatlas-base-dev \
        $FFMPEG_DEV \
        libcanberra-gtk3-module \
        libeigen3-dev \
        libglew-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstreamer1.0-dev \
        libgtk-3-dev \
        libjpeg-dev \
        libjpeg8-dev \
        libjpeg-turbo8-dev \
        libopenimageio-dev \
        openimageio-tools \
        libopenexr-dev \
        liblapack-dev \
        libopenblas-dev \
        libpng-dev \
        libtesseract-dev \
        libtiff-dev \
        libv4l-dev \
        libxine2-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtkglext1 \
        libgtkglext1-dev \
        pkg-config \
        qv4l2 \
        v4l-utils \
        zlib1g-dev \
        file \
        tar \
        $EXTRAS

# on x86, the python dev packages are already installed in the NGC containers under conda
# and installing them again from apt messes up their proper detection, so skip doing that
# these are needed however on other platforms (like aarch64) in order to build opencv-python
if [ $ARCH != "x86_64" ]; then
	echo "detected $ARCH, installing python3 dev packages..."

  if [[ $DISTRO != "24.04" ]]; then
    DIST_EXTRAS="python3-distutils python3-setuptools"
  fi

	apt-get install -y --no-install-recommends \
		python3-pip python3-dev $DIST_EXTRAS
fi

# restore cmake and numpy versions
bash /tmp/cmake/install.sh
bash /tmp/numpy/install.sh

rm -rf /var/lib/apt/lists/*
apt-get clean
