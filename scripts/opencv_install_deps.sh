#!/usr/bin/env bash
# this script installs build dependencies for compiling OpenCV

set -e -x

ARCH=$(uname -i)
echo "ARCH:  $ARCH"
	   
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
        libavresample-dev \
        libcanberra-gtk3-module \
        libdc1394-22-dev \
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
        liblapacke-dev \
        libopenblas-dev \
        libpng-dev \
        libpostproc-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
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
        zlib1g-dev

# on x86, the python dev packages are already installed in the NGC containers under conda
# and installing them again from apt messes up their proper detection, so skip doing that
# these are needed however on other platforms (like aarch64) in order to build opencv-python
if [ $ARCH != "x86_64" ]; then
	echo "detected $ARCH, installing python3 dev packages..."

	apt-get install -y --no-install-recommends \
		python3-pip \
		python3-dev \
		python3-numpy \
		python3-distutils \
		python3-setuptools
fi
