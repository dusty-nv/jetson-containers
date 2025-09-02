#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"
if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  wget $WGET_FLAGS \
  https://developer.download.nvidia.com/compute/cutensor/${CUTENSOR_VERSION}/local_installers/cutensor-local-repo-${DISTRO}-${CUTENSOR_VERSION}_${CUTENSOR_VERSION}-1_arm64.deb
elseif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
  echo 'Cutensor not supported by tegra (Jetson Orin)'
else
  wget $WGET_FLAGS \
  https://developer.download.nvidia.com/compute/cutensor/${CUTENSOR_VERSION}/local_installers/cutensor-local-repo-${DISTRO}-${CUTENSOR_VERSION}_${CUTENSOR_VERSION}-1_amd64.deb
fi
sudo dpkg -i cutensor-local-repo-${DISTRO}-${CUTENSOR_VERSION}_${CUTENSOR_VERSION}-1_*.deb
sudo cp /var/cutensor-local-repo-*-*/cutensor-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cutensor-cuda-${CUDA_VERSION_MAJOR}
rm -rf /var/lib/apt/lists/*
apt-get clean
