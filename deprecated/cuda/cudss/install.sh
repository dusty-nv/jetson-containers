#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"

# For JetPack 6/CUDA 12, use the network repo to avoid CUDA 13 dependencies in local-repo
if [ "${L4T_VERSION_MAJOR}" -ge 36 ] || [ "${CUDA_VERSION_MAJOR}" -eq 12 ]; then
  echo "Installing cuDSS via network repository..."
  if ! dpkg -l | grep -q "cuda-keyring"; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/arm64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
  fi
  apt-get update
  apt-get -y install --no-install-recommends cudss-cuda-12
else
  if [[ "$CUDA_ARCH" == "aarch64" || "$IS_SBSA" == "True" ]]; then
    wget $WGET_FLAGS \
    https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-${DISTRO}-${CUDSS_VERSION}_${CUDSS_VERSION}-1_arm64.deb
  elif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
    wget $WGET_FLAGS \
    https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-tegra-repo-${DISTRO}-${CUDSS_VERSION}_${CUDSS_VERSION}-1_arm64.deb
  else
    wget $WGET_FLAGS \
    https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-${DISTRO}-${CUDSS_VERSION}_*-1_amd64.deb
  fi
  dpkg -i cudss-local-*.deb
  cp /var/cudss-local-*/cudss-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install cudss
fi

rm -rf /var/lib/apt/lists/*
apt-get clean
rm -rf /tmp/*.deb
rm -rf /*.deb
echo "cuDSS ${CUDSS_VERSION} installed successfully"
