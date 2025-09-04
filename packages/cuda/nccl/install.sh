#!/usr/bin/env bash
set -euxo pipefail

echo "Installing NVIDIA NCCL $NCCL_VERSION"

if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  DEB="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda13.0_1.0-1_arm64.deb"

elif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
  if [[ "$FORCE_BUILD" == "on" ]]; then
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
if [[ "$CUDA_ARCH" != "tegra-aarch64" ]]; then
    dpkg -i $DEB
    sudo cp /var/nccl-local-repo-ubuntu2404-2.27.7-cuda13.0/nccl-local-190A5319-keyring.gpg /usr/share/keyrings/
    apt-get update
    dpkg -i $DEB
    apt-get -y install libnccl2 libnccl-dev
else
cd "$TMP"
wget $WGET_FLAGS "$MULTIARCH_URL/$DEB"
dpkg -i "$DEB"
cp /var/nccl-local-repo-ubuntu2404-"$NCCL_VERSION"-cuda13.0/nccl-local-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install libnccl2 libnccl-dev
fi
