#!/usr/bin/env bash
set -ex

ARCH=$(uname -m)
ARCH_TYPE=$ARCH

# Detectar si es Tegra
if [[ "$ARCH" == "aarch64" ]]; then
    if uname -a | grep -qi tegra; then
        ARCH_TYPE="tegra-aarch64"
    fi
fi

echo "Detected architecture: ${ARCH_TYPE}"

apt-get update
apt-get install -y --no-install-recommends \
        binutils \
        xz-utils
rm -rf /var/lib/apt/lists/*
apt-get clean

echo "Downloading ${CUDA_DEB}"
mkdir -p /tmp/cuda
cd /tmp/cuda

if [[ "$ARCH_TYPE" == "tegra-aarch64" ]]; then
    # Jetson (Tegra)
    wget --quiet --show-progress --progress=bar:force:noscroll \
        https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/arm64/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600
else
    # ARM64 SBSA (Grace)
    wget --quiet --show-progress --progress=bar:force:noscroll \
        https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/sbsa/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600
fi
wget --quiet --show-progress --progress=bar:force:noscroll ${CUDA_URL}

dpkg -i *.deb

cp /var/cuda-*-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Tegra (Jetson)
if [[ "$ARCH_TYPE" == "tegra-aarch64" ]]; then
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