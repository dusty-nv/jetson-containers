#!/usr/bin/env bash
set -ex

# Determine architecture for the repository
ARCH=$(dpkg --print-architecture)
echo "Setting up NVIDIA repository for ${ARCH} on ${DISTRO}"

# Download and install the cuda-keyring
# This is the modern way to manage NVIDIA repos (Network Repos)
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"

echo "Downloading keyring from ${KEYRING_URL}..."
wget --quiet ${KEYRING_URL} -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb

# Update APT lists to include the new repository
apt-get update

echo "NVIDIA Network Repository (cuda-repo) configured successfully."
