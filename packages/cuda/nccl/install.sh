#!/usr/bin/env bash
set -ex
echo "Installing NVIDIA NCCL $NCCL_VERSION (NCCL)"
if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  DEB="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda13.0_1.0-1_arm64.deb"
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
fi
