#!/usr/bin/env bash
# Install cuDNN
set -eux

echo "Installing cuDNN ${CUDNN_VERSION}..."

cd ${TMP:-/tmp}

# Download cuDNN .deb file
wget ${WGET_FLAGS} ${CUDNN_URL}

# Install the repository package
dpkg -i *.deb

# Copy keyring
cp /var/cudnn-*-repo-*/cudnn-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true
cp /var/cudnn-*-repo-*-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true


# Update and install cuDNN packages
apt-get update
apt-get install -y --no-install-recommends ${CUDNN_PACKAGES}

# Cleanup
dpkg -P ${CUDNN_DEB} 2>/dev/null || true
rm -rf /var/lib/apt/lists/*
apt-get clean
rm -rf /tmp/*.deb
rm -rf /*.deb
echo "cuDNN ${CUDNN_VERSION} installed successfully"
