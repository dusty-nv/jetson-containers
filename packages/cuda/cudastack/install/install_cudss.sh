#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"
echo "Detected CUDA Version (Major): ${CUDA_VERSION_MAJOR}"
echo "Detected L4T Version (Major): ${L4T_VERSION_MAJOR}"

# cuDSS 0.7.1 local installer (arm64) depends on libcublas.so.13, which is not in JP 6.1
# For JP 6 (CUDA 12), we ALWAYS use the network repository configured by cuda-repo
if [ "${L4T_VERSION_MAJOR}" -ge 36 ] || [ "${CUDA_VERSION_MAJOR}" -eq 12 ]; then
  echo "Installing cuDSS via network repository (cudss-cuda-12)..."
  # apt-get update is already done by cuda-repo, but we can do it to be safe
  apt-get update
  apt-get -y install --no-install-recommends cudss-cuda-12
else
  # Fallback to local repo for older versions (JetPack 5/CUDA 11)
  if [ "$CUDA_ARCH" = "aarch64" ] || [ "$IS_SBSA" = "True" ]; then
    wget $WGET_FLAGS \
    https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-${DISTRO}-${CUDSS_VERSION}_${CUDSS_VERSION}-1_arm64.deb
  elif [ "$CUDA_ARCH" = "tegra-aarch64" ]; then
    # In older JP (L4T < 36), use the tegra-specific repo if available
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
