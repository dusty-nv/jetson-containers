#!/usr/bin/env bash
set -exu

echo "Detected architecture: ${CUDA_ARCH}"

if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/nvshmem/${NVSHMEM_VERSION}/local_installers/nvshmem-local-repo-${DISTRO}-${NVSHMEM_VERSION}_${NVSHMEM_VERSION}-1_arm64.deb"
  dpkg -i nvshmem-local-*.deb
  cp /var/nvshmem-local-*/nvshmem-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install nvshmem

elif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
  # Install from .tar.xz for Jetson Orin @ 22.04 and 24.04
#  wget $WGET_FLAGS "https://developer.download.nvidia.com/compute/nvshmem/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-${NVSHMEM_VERSION}.0-archive.tar.xz"
#  tar -xJf libcusparse_lt-linux-aarch64-${NVSHMEM_VERSION}.0-archive.tar.xz
#  cd libcusparse_lt-linux-aarch64-${NVSHMEM_VERSION}.0-archive

#  if ls *.deb 1>/dev/null 2>&1; then
#    dpkg -i *.deb || apt-get -f install -y
##  else
 #   cp -v lib/* /usr/lib/aarch64-linux-gnu/ || cp -v lib/* /usr/local/cuda/lib64/
#    cp -v include/* /usr/local/cuda/include/
    
#    ldconfig -p | grep nvshmem  || true
#  fi
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/nvshmem/${NVSHMEM_VERSION}/local_installers/nvshmem-local-repo-${DISTRO}-${NVSHMEM_VERSION}_${NVSHMEM_VERSION}-1_arm64.deb"
  dpkg -i nvshmem-local-*.deb
  cp /var/nvshmem-local-*/nvshmem-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install nvshmem

else
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/nvshmem/${NVSHMEM_VERSION}/local_installers/nvshmem-local-repo-${DISTRO}-${NVSHMEM_VERSION}_${NVSHMEM_VERSION}-1_amd64.deb"
  
  dpkg -i nvshmem-local-*.deb
  cp /var/nvshmem-local-*/nvshmem-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install nvshmem
fi

# Clean up (only apt cache; won't touch manual lib copies)
rm -rf /var/lib/apt/lists/*
apt-get clean
