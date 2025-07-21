#!/usr/bin/env bash
set -ex

echo "Installing NVIDIA GDRCopy $GDRCOPY_VERSION (GDRCopy)"
git clone --recursive --branch=cdmm https://github.com/NVIDIA/gdrcopy.git /opt/gdrcopy
cd /opt/gdrcopy/packages
sed -i 's/gdrdrv-dkms (= @FULL_VERSION@), //g' debian-meta/control
CUDA=/usr/local/cuda ./build-deb-packages.sh
mkdir debs
mv *.deb debs
rm -rf debs/gdrdrv-dkms*.deb
ls -la debs
dpkg -i debs/*.deb
rm -rf /opt/gdrcopy
ldconfig
