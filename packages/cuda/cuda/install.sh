
#!/usr/bin/env bash
set -ex

IS_SBSA=$([ -z "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep nvgpu)" ] && echo 1 || echo 0)

apt-get update
apt-get install -y --no-install-recommends \
        binutils \
        xz-utils
rm -rf /var/lib/apt/lists/*
apt-get clean

echo "Downloading ${CUDA_DEB}"
mkdir -p /tmp/cuda
cd /tmp/cuda

if [[ -z $IS_SBSA ]]; then
    # Jetson (Tegra)
    wget $WGET_FLAGS \
        https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/arm64/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600
else
    # ARM64 SBSA (Grace)
    wget $WGET_FLAGS \
        https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/sbsa/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600
fi

wget $WGET_FLAGS ${CUDA_URL}
dpkg -i *.deb
cp /var/cuda-*-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Tegra (Jetson)
if [[ -z $IS_SBSA ]]; then
    ar x /var/cuda-tegra-repo-ubuntu*-local/cuda-compat-*.deb
    tar xvf data.tar.xz -C /
fi

apt-get update
apt-get install -y --no-install-recommends ${CUDA_PACKAGES}

# if ARM64 SBSA (Grace)
if [[ ! -z $IS_SBSA ]]; then
    sudo apt-get -y --no-install-recommends install nvpl
fi

rm -rf /var/lib/apt/lists/*
apt-get clean

dpkg --list | grep cuda
dpkg -P ${CUDA_DEB}
rm -rf /tmp/cuda
