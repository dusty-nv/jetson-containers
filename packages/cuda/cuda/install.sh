#!/usr/bin/env bash
# CUDA Toolkit installer
set -ex

apt-get update
apt-get install -y --no-install-recommends \
        binutils \
        xz-utils
rm -rf /var/lib/apt/lists/*
apt-get clean

echo "Downloading ${CUDA_DEB}"
mkdir -p /tmp/cuda
cd /tmp/cuda

wget --quiet --show-progress --progress=bar:force:noscroll https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/arm64/cuda-${DISTRO}.pin -O /etc/apt/preferences.d/cuda-repository-pin-600
wget --quiet --show-progress --progress=bar:force:noscroll ${CUDA_URL}

dpkg -i *.deb

cp /var/cuda-*-local/cuda-*-keyring.gpg /usr/share/keyrings/

if [ "$(uname -m)" = "aarch64" ]; then
   ar x /var/cuda-tegra-repo-ubuntu*-local/cuda-compat-*.deb
   tar xvf data.tar.xz -C /
fi

apt-get update
apt-get install -y --no-install-recommends ${CUDA_PACKAGES}
rm -rf /var/lib/apt/lists/*
apt-get clean

dpkg --list | grep cuda
dpkg -P ${CUDA_DEB}
rm -rf /tmp/cuda
   