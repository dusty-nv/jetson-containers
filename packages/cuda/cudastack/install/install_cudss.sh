#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"
echo "Detected CUDA Version (Major): ${CUDA_VERSION_MAJOR}"
echo "Detected L4T Version (Major): ${L4T_VERSION_MAJOR}"

# cuDSS 0.7.1 local installer (arm64) depends on libcublas.so.13, which is not in JP 6.1
# To ensure build determinism and ABI compatibility on JP 6 (CUDA 12), 
# we download explicit .deb artifacts from the NVIDIA network repository pool.
if [ "${L4T_VERSION_MAJOR}" -ge 36 ] || [ "${CUDA_VERSION_MAJOR}" -eq 12 ]; then
  echo "Installing cuDSS via explicit network artifacts for determinism..."
  
  # Determine architecture for URL substitution (arm64 or sbsa)
  ARCH=$(dpkg --print-architecture)
  
  # Substitute placeholders in the URLs (if any)
  # CUDSS_URL_JETPACK6 may contain multiple URLs separated by space
  DISTRO=${DISTRO:-$(lsb_release -cs)}
  WGET_FLAGS=${WGET_FLAGS:-"--quiet"}

  for URL in ${CUDSS_URL_JETPACK6//"{distro}"/"$DISTRO"}; do
    URL_FINAL=${URL//"{arch}"/"$ARCH"}
    echo "Downloading ${URL_FINAL}..."
    wget --quiet ${URL_FINAL} -P /tmp/cudss_debs/
  done
  
  dpkg -i /tmp/cudss_debs/*.deb
  rm -rf /tmp/cudss_debs/
else
  # Fallback to local repo for older versions (JetPack 5/CUDA 11)
  if [ "$CUDA_ARCH" = "aarch64" ] || [ "$IS_SBSA" = "True" ]; then
    wget $WGET_FLAGS \
    https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-${DISTRO}-${CUDSS_VERSION}_${CUDSS_VERSION}-1_arm64.deb
  elif [ "$CUDA_ARCH" = "tegra-aarch64" ]; then
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
