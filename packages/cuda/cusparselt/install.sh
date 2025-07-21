#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"
if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  wget $WGET_FLAGS \
  https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_1.0-1_aarch64.deb
elif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
  wget $WGET_FLAGS https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/cusparselt-local-tegra-repo-${DISTRO}-${CUSPARSELT_VERSION}_1.0-1_arm64.deb
else
  wget $WGET_FLAGS \
  https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_1.0-1_amd64.deb
fi
  dpkg -i cusparselt-local-*.deb
  cp /var/cusparselt-local-*/cusparselt-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install libcusparselt0 libcusparselt-dev
fi
rm -rf /var/lib/apt/lists/*
apt-get clean
