#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"
# ARM64 SBSA (Grace) NVIDIA Performance Libraries (NVPL)
# NVPL allows you to easily port HPC applications to NVIDIA Grace™ CPU platforms to achieve industry-leading performance and efficiency.
if [[ "$CUDA_ARCH" == "aarch64" ]]; then
    apt-get update
    apt-get -y install nvpl
fi
rm -rf /var/lib/apt/lists/*
apt-get clean

dpkg --list | grep cuda
dpkg -P ${CUDA_DEB}
rm -rf /tmp/cuda
