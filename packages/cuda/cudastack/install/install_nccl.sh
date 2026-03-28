#!/usr/bin/env bash
set -eux

echo "Installing NVIDIA NCCL $NCCL_VERSION via network repo"

if [ "$CUDA_ARCH" = "tegra-aarch64" ] && [ "${CUDA_INSTALLED_VERSION}" -lt 132 ]; then
    REPO_ARCH="arm64"
elif [ "$(uname -m)" = "aarch64" ]; then
    REPO_ARCH="sbsa"
else
    REPO_ARCH="x86_64"
fi

CUDA_MAJOR="${CUDA_INSTALLED_VERSION:0:2}"
CUDA_MINOR="${CUDA_INSTALLED_VERSION:2}"
CUDA_VER="${CUDA_MAJOR}.${CUDA_MINOR}"
NCCL_APT_VER="${NCCL_VERSION}-1+cuda${CUDA_VER}"

echo "NCCL apt version: ${NCCL_APT_VER} (repo: ${DISTRO}/${REPO_ARCH})"

cd /tmp

wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb" \
    -O cuda-keyring.deb
dpkg -i cuda-keyring.deb
apt-get update

apt-get install -y --no-install-recommends \
    libnccl2=${NCCL_APT_VER} \
    libnccl-dev=${NCCL_APT_VER}

# Remove cuda-keyring to prevent the global NVIDIA repo from propagating
# to subsequent build layers (preserves version control of installed packages)
dpkg --purge cuda-keyring 2>/dev/null || true
rm -f /etc/apt/sources.list.d/cuda-*-keyring.list
rm -f /etc/apt/preferences.d/cuda-repository-pin-600
rm -f /usr/share/keyrings/cuda-archive-keyring.gpg

rm -f /tmp/cuda-keyring.deb
rm -rf /var/lib/apt/lists/*
apt-get clean

echo "NVIDIA NCCL $NCCL_VERSION installed successfully"
