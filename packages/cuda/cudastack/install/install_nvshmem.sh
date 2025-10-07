#!/usr/bin/env bash
set -exu

echo "Detected architecture: ${CUDA_ARCH}"

if [ "$CUDA_ARCH" = "aarch64" ] || [ "$CUDA_ARCH" = "tegra-aarch64" ]; then
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/nvshmem/${NVSHMEM_VERSION}/local_installers/nvshmem-local-repo-${DISTRO}-${NVSHMEM_VERSION}_${NVSHMEM_VERSION}-1_arm64.deb"
  dpkg -i nvshmem-local-*.deb
  cp /var/nvshmem-local-*/nvshmem-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install nvshmem-cuda-${CUDA_VERSION_MAJOR}

# elif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
# echo "Not supported on Tegra architecture"
else
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/nvshmem/${NVSHMEM_VERSION}/local_installers/nvshmem-local-repo-${DISTRO}-${NVSHMEM_VERSION}_${NVSHMEM_VERSION}-1_amd64.deb"

  dpkg -i nvshmem-local-*.deb
  cp /var/nvshmem-local-*/nvshmem-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install nvshmem-cuda-${CUDA_VERSION_MAJOR}
fi

# Clean up (only apt cache; won't touch manual lib copies)
rm -rf /var/lib/apt/lists/*
apt-get clean
rm -rf /tmp/*.deb
rm -rf /*.deb
echo "NVSHMEM ${NVSHMEM_VERSION} installed successfully"
