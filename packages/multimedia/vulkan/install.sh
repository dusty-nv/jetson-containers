#!/usr/bin/env bash
set -ex

# skip build targets: libvulkan1 libvulkan-dev vulkan-tools
apt-get update
apt-get install -y --no-install-recommends \
    libglm-dev libxcb-dri3-0 libxcb-present0 libpciaccess0 \
    libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev \
    libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev \
    python-is-python3 bison libx11-xcb-dev liblz4-dev libzstd-dev \
    ocaml-core ninja-build pkg-config libxml2-dev wayland-protocols \
    qtbase5-dev qt6-base-dev python3-jsonschema
    
bash /tmp/cmake/install.sh # restore cmake
rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of Vulkan SDK $VULKAN_VERSION"
	exit 1
fi

tarpack install vulkan-sdk-$VULKAN_VERSION
echo "installed" > "$TMP/.vulkan"
ldconfig