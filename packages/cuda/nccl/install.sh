#!/usr/bin/env bash
set -ex
echo "Installing NVIDIA NCCL $NCCL_VERSION (NCCL)"
if [[ "$CUDA_ARCH" == "aarch64" ]]; then
  DEB="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda12.9_1.0-1_aarch64.deb"
else
  DEB="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda12.9_1.0-1_amd64.deb"
fi
cd $TMP
if [ ! -f $TXZ ]; then
  wget $WGET_FLAGS $MULTIARCH_URL/$DEB
fi
if [[ "$CUDA_ARCH" != "tegra-aarch64" ]]; then
    dpkg -i $DEB
    apt-get update
    dpkg -i $DEB
    apt-get -y install libnccl2 libnccl-dev
fi
