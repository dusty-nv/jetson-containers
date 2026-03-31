#!/usr/bin/env bash
set -eux

echo "Detected architecture: ${CUDA_ARCH}"

CUDA_MAJOR=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p')
: "${CUDA_MAJOR:=${CUDA_VERSION_MAJOR:-12}}"

if [ "$(uname -m)" = "aarch64" ]; then
    DEB_ARCH="arm64"
else
    DEB_ARCH="amd64"
fi

DEB="cutensor-local-repo-${DISTRO}-${CUTENSOR_VERSION}_${CUTENSOR_VERSION}-1_${DEB_ARCH}.deb"
URL="https://developer.download.nvidia.com/compute/cutensor/${CUTENSOR_VERSION}/local_installers/${DEB}"

cd /tmp
echo "Downloading cuTENSOR ${CUTENSOR_VERSION} local repo from ${URL}"
wget ${WGET_FLAGS:-} "${URL}" -O "${DEB}"
dpkg -i "${DEB}"
cp /var/cutensor-local-repo-*/cutensor-*-keyring.gpg /usr/share/keyrings/
apt-get update

apt-get install -y --no-install-recommends "cutensor-cuda-${CUDA_MAJOR}"

# Cleanup local repo and apt caches
rm -rf /var/cutensor-local-repo-*
rm -rf /etc/apt/sources.list.d/*
rm -rf /var/lib/apt/lists/*
apt-get clean
rm -rf /tmp/*.deb
rm -rf /*.deb

echo "cuTENSOR ${CUTENSOR_VERSION} installed successfully"
