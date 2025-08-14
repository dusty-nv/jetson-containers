#!/usr/bin/env bash
set -exu

echo "Detected architecture: ${CUDA_ARCH}"

if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  # wget $WGET_FLAGS \
  #  "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_1.0-1_arm64.deb"
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/cusparselt-local-repo-${DISTRO}-0.8.0_1.0-1_arm64.deb"
  dpkg -i cusparselt-local-*.deb
  cp /var/cusparselt-local-*/cusparselt-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install libcusparselt0 libcusparselt-dev

elif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
  # Install from .tar.xz for Jetson Orin @ 22.04 and 24.04
  wget $WGET_FLAGS "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}.0-archive.tar.xz"
  tar -xJf libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}.0-archive.tar.xz
  cd libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}.0-archive

  if ls *.deb 1>/dev/null 2>&1; then
    dpkg -i *.deb || apt-get -f install -y
  else
    cp -v lib/* /usr/lib/aarch64-linux-gnu/ || cp -v lib/* /usr/local/cuda/lib64/
    cp -v include/* /usr/local/cuda/include/
    
    ldconfig -p | grep libcusparseLt || true
  fi

else
  # wget $WGET_FLAGS \
  #  "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_1.0-1_amd64.deb"
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/cusparselt/0.8.0/local_installers/cusparselt-local-repo-${DISTRO}-0.8.0_1.0-1_amd64.deb"
  
  dpkg -i cusparselt-local-*.deb
  cp /var/cusparselt-local-*/cusparselt-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install libcusparselt0 libcusparselt-dev
fi

# Clean up (only apt cache; won't touch manual lib copies)
rm -rf /var/lib/apt/lists/*
apt-get clean
