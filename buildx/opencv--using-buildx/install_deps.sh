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
  EXTRAS="libtbbmalloc2 libtbb-dev $EXTRAS"
else
  EXTRAS="libtbb2 libtbb2-dev liblapacke-dev $EXTRAS"
fi

apt-get update
apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        cmake \
        git \
        file \
        tar \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
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
        liblapack-dev \
        libopenblas-dev \
        libpng-dev \
        libpostproc-dev \
        libswscale-dev \
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
		python3-pip \
		python3-dev \
    $DIST_EXTRAS

	python3 -c 'import numpy; print("NumPy version before installation:", numpy.__version__)' 2>/dev/null

  if [ $? != 0 ]; then
      echo "NumPy not found. Installing NumPy 2.0..."
      apt-get update
      # apt-get install -y --no-install-recommends python3-numpy
      python3 -m pip install "numpy>=2.0.0" --break-system-packages
  fi
fi

rm -rf /var/lib/apt/lists/*
apt-get clean
