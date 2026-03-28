#!/usr/bin/env bash
set -eux

echo "Detected architecture: ${CUDA_ARCH}"

if [ "$CUDA_ARCH" = "tegra-aarch64" ] && [ "${CUDA_INSTALLED_VERSION}" -lt 132 ]; then
    REPO_ARCH="arm64"
elif [ "$(uname -m)" = "aarch64" ]; then
    REPO_ARCH="sbsa"
else
    REPO_ARCH="x86_64"
fi

cd /tmp

wget ${WGET_FLAGS:-} \
    "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb" \
    -O cuda-keyring.deb
dpkg -i cuda-keyring.deb
apt-get update

PKG="cutensor-cuda-${CUDA_VERSION_MAJOR}"

# Find exact apt version matching CUTENSOR_VERSION and pin to it
CUTENSOR_APT_VER=$(apt-cache madison "${PKG}" 2>/dev/null \
    | awk -v ver="${CUTENSOR_VERSION}" '$3 ~ ver {gsub(/^ +| +$/, "", $3); print $3; exit}')

if [ -n "${CUTENSOR_APT_VER}" ]; then
    echo "Pinning cuTENSOR to: ${PKG}=${CUTENSOR_APT_VER}"
    apt-get install -y --no-install-recommends "${PKG}=${CUTENSOR_APT_VER}"
else
    echo "Exact version ${CUTENSOR_VERSION} not found in repo, installing latest ${PKG}"
    apt-get install -y --no-install-recommends "${PKG}"
fi

# Remove cuda-keyring to prevent the global NVIDIA repo from propagating
dpkg --purge cuda-keyring 2>/dev/null || true
rm -f /etc/apt/sources.list.d/cuda-*-keyring.list
rm -f /etc/apt/preferences.d/cuda-repository-pin-600
rm -f /usr/share/keyrings/cuda-archive-keyring.gpg

rm -f /tmp/cuda-keyring.deb
rm -rf /var/lib/apt/lists/*
apt-get clean
rm -rf /tmp/*.deb
rm -rf /*.deb

echo "cuTENSOR ${CUTENSOR_VERSION} installed successfully"
