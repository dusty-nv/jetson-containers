#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"
# ARM64 SBSA (Grace) NVIDIA Performance Libraries (NVPL)
# NVPL allows you to easily port HPC applications to NVIDIA Grace™ CPU platforms to achieve industry-leading performance and efficiency.
if [[ "$CUDA_ARCH" == "aarch64" || "$IS_SBSA" == "True" ]]; then
    wget $WGET_FLAGS https://developer.download.nvidia.com/compute/nvpl/${NVPL_VERSION}/local_installers/nvpl-local-repo-${DISTRO}-${NVPL_VERSION}_1.0-1_arm64.deb
    sudo dpkg -i nvpl-local-*.deb
    sudo cp /var/nvpl-local-repo-*-*/nvpl-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install nvpl
fi
rm -rf /var/lib/apt/lists/*
apt-get clean
