#!/usr/bin/env bash
# eudev
set -xo pipefail

# TODO: Create build & install sh scripts with package upload

# Install dependencies
apt-get update
apt-get install -y --no-install-recommends \
	autoconf \
    automake \
    docbook-xml \
    docbook-xsl \
    gnu-efi \
    gperf \
    intltool \
    libacl1-dev \
    libblkid-dev \
    libcap-dev \
    libkmod-dev \
    libselinux1-dev \
    libtool \
    m4 \
    pkg-config \
    xsltproc
apt-get clean
rm -rf /var/lib/apt/lists/*

wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate \
    https://github.com/eudev-project/eudev/releases/download/${EUDEV_VERSION}/eudev-${EUDEV_VERSION#v}.tar.gz \
    -O /tmp/eudev.tar.gz
tar -xzvf /tmp/eudev.tar.gz -C /tmp

cd /tmp/eudev-${EUDEV_VERSION#v}
ls -l .
./configure --prefix=/usr
make
make install

dpkg -l | grep udev
which udevd

udev --version
udevd --version

rm -rf /tmp/eudev
