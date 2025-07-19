#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"

apt-get update
apt-get install -y --no-install-recommends \
        binutils \
        xz-utils
rm -rf /var/lib/apt/lists/*
apt-get clean

echo "Downloading ${CUDA_DEB}"
mkdir -p /tmp/cuda
cd /tmp/cuda

if [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
    # Jetson (Tegra)
    wget $WGET_FLAGS \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-ubuntu2204.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600

elif [[ "$CUDA_ARCH" == "aarch64" ]]; then
    # ARM64 SBSA (Grace)
    wget $WGET_FLAGS \
        https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/sbsa/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600
else
    # x86_64
    wget $WGET_FLAGS \
        https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600
fi

wget $WGET_FLAGS ${CUDA_URL}
dpkg -i *.deb
cp /var/cuda-*-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Tegra (Jetson)
if [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
    ar x /var/cuda-tegra-repo-ubuntu*-local/cuda-compat-*.deb
    tar xvf data.tar.xz -C /
fi

apt-get update
apt-get install -y --no-install-recommends ${CUDA_PACKAGES}

if [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
    # Jetson (Tegra)
    wget $WGET_FLAGS \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb

elif [[ "$CUDA_ARCH" == "aarch64" ]]; then
    # ARM64 SBSA (Grace)
    wget $WGET_FLAGS \
        https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/sbsa/cuda-keyring_1.1-1_all.deb
else
    # x86_64
    wget $WGET_FLAGS \
        https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb
fi
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update && \
apt-get install -y --no-install-recommends libcusparselt0 libcusparselt-dev

if [[ "$CUDA_ARCH" != "tegra-aarch64" ]]; then
    apt-get install -y --no-install-recommends libcutensor2 libcutensor-dev libcutensor-doc
fi
rm -rf /var/lib/apt/lists/*
apt-get clean

dpkg --list | grep cuda
dpkg -P ${CUDA_DEB}
rm -rf /tmp/cuda
