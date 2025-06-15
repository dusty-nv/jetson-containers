#!/usr/bin/env bash
echo "Installing NVIDIA Warp $WARP_VERSION"
set -ex
cd $TMP

WARP_REPO="${WARP_REPO:=https://github.com/NVIDIA/warp}"

pip3 install usd-core matplotlib pyglet

if [ $WARP_INSTALL == "nightly" ]; then
  echo "install nightly"
else
  wget $WGET_FLAGS $WARP_INSTALL
  pip3 install *.whl
fi

git clone --recursive --depth=1 $WARP_REPO $SRC -b v$WARP_VERSION || \
git clone --recursive --depth=1 $WARP_REPO $SRC
