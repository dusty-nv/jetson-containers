#!/usr/bin/env bash
set -eux

echo "Installing NVIDIA NCCL $NCCL_VERSION"

if [ "$CUDA_ARCH" = "aarch64" ]; then
  DEB="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda13.0_1.0-1_arm64.deb"

elif [ "$CUDA_ARCH" = "tegra-aarch64" ]; then
  if [ "$FORCE_BUILD" = "on" ]; then
    echo "Forcing build of NVIDIA NCCL ${NCCL_VERSION}"
    exit 1
  fi
  tarpack install "nccl-${NCCL_VERSION}"
  exit 0
else
  DEB="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda13.0_1.0-1_amd64.deb"
fi

cd $TMP
wget $WGET_FLAGS $MULTIARCH_URL/$DEB
if [ "$CUDA_ARCH" != "tegra-aarch64" ]; then
    dpkg -i $DEB
    cp /var/nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda13.0/nccl-local-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    dpkg -i $DEB
    apt-get -y install libnccl2 libnccl-dev
else
cd "$TMP"
wget $WGET_FLAGS "$MULTIARCH_URL/$DEB"
dpkg -i "$DEB"
cp /var/nccl-local-repo-${DISTRO}-"$NCCL_VERSION"-cuda13.0/nccl-local-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install libnccl2 libnccl-dev
fi
rm -rf /tmp/*.deb
rm -rf /*.deb
rm -rf /var/lib/apt/lists/*
apt-get clean
echo "NVIDIA NCCL $NCCL_VERSION installed successfully"
