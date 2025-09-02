#!/usr/bin/env bash
set -euo pipefail
set -x

echo "Detected architecture: ${CUDA_ARCH}"

if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  deb="cutensor-local-repo-${DISTRO}-${CUTENSOR_VERSION}_${CUTENSOR_VERSION}-1_arm64.deb"
elif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
  echo 'cuTENSOR is not supported on Tegra (Jetson Orin)'
  exit 0
else
  deb="cutensor-local-repo-${DISTRO}-${CUTENSOR_VERSION}_${CUTENSOR_VERSION}-1_amd64.deb"
fi

wget ${WGET_FLAGS:-} "https://developer.download.nvidia.com/compute/cutensor/${CUTENSOR_VERSION}/local_installers/${deb}"
sudo dpkg -i "$deb"
sudo cp /var/cutensor-local-repo-*-*/cutensor-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install "cutensor-cuda-${CUDA_VERSION_MAJOR}"
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get clean
