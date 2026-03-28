#!/usr/bin/env bash
set -exu

echo "Detected architecture: ${CUDA_ARCH}"

CUDA_MAJOR=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p')
: "${CUDA_MAJOR:=${CUDA_VERSION_MAJOR:-13}}"

if [ "$CUDA_ARCH" = "tegra-aarch64" ] && [ "${CUDA_INSTALLED_VERSION}" -lt 132 ]; then
    REPO_ARCH="arm64"
elif [ "$(uname -m)" = "aarch64" ]; then
    REPO_ARCH="sbsa"
else
    REPO_ARCH="x86_64"
fi

cd /tmp
wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb" \
    -O cuda-keyring.deb
dpkg -i cuda-keyring.deb
apt-get update

PKG_RT="libcusparselt0-cuda-${CUDA_MAJOR}"
PKG_DEV="libcusparselt0-dev-cuda-${CUDA_MAJOR}"
if ! apt-cache show "${PKG_RT}" >/dev/null 2>&1; then
    PKG_RT="libcusparselt0"
    PKG_DEV="libcusparselt0-dev"
fi

RT_APT_VER=$(apt-cache madison "${PKG_RT}" 2>/dev/null \
    | awk -v ver="${CUSPARSELT_VERSION}" '$3 ~ ver {gsub(/^ +| +$/, "", $3); print $3; exit}')
DEV_APT_VER=$(apt-cache madison "${PKG_DEV}" 2>/dev/null \
    | awk -v ver="${CUSPARSELT_VERSION}" '$3 ~ ver {gsub(/^ +| +$/, "", $3); print $3; exit}')

if [ -n "${RT_APT_VER}" ] && [ -n "${DEV_APT_VER}" ]; then
    echo "Pinning cuSPARSELt to: ${PKG_RT}=${RT_APT_VER} ${PKG_DEV}=${DEV_APT_VER}"
    apt-get install -y --no-install-recommends "${PKG_RT}=${RT_APT_VER}" "${PKG_DEV}=${DEV_APT_VER}"
else
    echo "Exact version ${CUSPARSELT_VERSION} not found in repo, installing latest"
    apt-get install -y --no-install-recommends "${PKG_RT}" "${PKG_DEV}"
fi

ldconfig

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

echo "cuSPARSELt ${CUSPARSELT_VERSION} installed successfully"
