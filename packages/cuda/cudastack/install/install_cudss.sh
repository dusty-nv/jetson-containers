#!/usr/bin/env bash
set -ex
echo "Detected architecture: ${CUDA_ARCH}"
if [ "$CUDA_ARCH" = "aarch64" ] || [ "$IS_SBSA" = "True" ]; then
  wget $WGET_FLAGS \
  https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-${DISTRO}-${CUDSS_VERSION}_${CUDSS_VERSION}-1_arm64.deb
elif [ "$CUDA_ARCH" = "tegra-aarch64" ]; then
  if [ "${CUDA_INSTALLED_VERSION}" -ge 130 ]; then
    wget $WGET_FLAGS \
    https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-${DISTRO}-${CUDSS_VERSION}_${CUDSS_VERSION}-1_arm64.deb
  else
    wget $WGET_FLAGS \
    https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-tegra-repo-${DISTRO}-${CUDSS_VERSION}_${CUDSS_VERSION}-1_arm64.deb
  fi
else
  wget $WGET_FLAGS \
  https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-${DISTRO}-${CUDSS_VERSION}_*-1_amd64.deb
fi
dpkg -i cudss-local-*.deb
cp /var/cudss-local-*/cudss-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudss
rm -rf /var/lib/apt/lists/*
apt-get clean
rm -rf /tmp/*.deb
rm -rf /*.deb
echo "cuDSS ${CUDSS_VERSION} installed successfully"
