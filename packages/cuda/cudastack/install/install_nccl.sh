#!/usr/bin/env bash
set -eux

echo "Installing NVIDIA NCCL $NCCL_VERSION"

if [ -z "${NCCL_URL:-}" ]; then
    echo "No NCCL_URL provided, falling back to build from source"
    exit 1
fi

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    LIB_DIR="/usr/lib/aarch64-linux-gnu"
else
    LIB_DIR="/usr/lib/x86_64-linux-gnu"
fi

cd "$TMP"
wget $WGET_FLAGS "$NCCL_URL" -O nccl.txz
tar -xf nccl.txz

NCCL_DIR=$(ls -d nccl_* | head -1)

cp -a "$NCCL_DIR"/include/* /usr/include/
cp -a "$NCCL_DIR"/lib/* "$LIB_DIR"/
ldconfig

rm -rf nccl.txz "$NCCL_DIR"

echo "NVIDIA NCCL $NCCL_VERSION installed successfully"
