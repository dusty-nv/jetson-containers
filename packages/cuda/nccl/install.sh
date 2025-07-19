#!/usr/bin/env bash
set -ex

echo "Detected architecture: ${CUDA_ARCH}"
# NCCL
if [[ "$CUDA_ARCH" != "tegra-aarch64" ]]; then
    apt-get update
    apt-get -y install libnccl2 libnccl-dev
fi
rm -rf /var/lib/apt/lists/*
apt-get clean
