#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"
if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  wget $WGET_FLAGS \
  https://developer.download.nvidia.com/compute/cutensor/${CUTENSOR_VERSION}/local_installers/cutensor-local-repo-${DISTRO}-${CUTENSOR_VERSION}_1.0-1_aarch64.deb
else
  wget $WGET_FLAGS \
  https://developer.download.nvidia.com/compute/cutensor/${CUTENSOR_VERSION}/local_installers/cutensor-local-repo-${DISTRO}-${CUTENSOR_VERSION}_1.0-1_amd64.deb
fi
  sudo dpkg -i cutensor-local-repo-*-*_1.0-1_amd64.deb
  sudo cp /var/cutensor-local-repo-*-*/cutensor-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install libcutensor2 libcutensor-dev libcutensor-doc
fi
rm -rf /var/lib/apt/lists/*
apt-get clean
