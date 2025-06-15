#!/usr/bin/env bash
printf "\nDownloading Vulkan SDK $VULKAN_VERSION\n"
set -ex

cd $TMP
wget $WGET_FLAGS https://sdk.lunarg.com/sdk/download/$VULKAN_VERSION/linux/vulkansdk-linux-x86_64-$VULKAN_VERSION.tar.xz
tar xvf vulkansdk-*.tar.xz

cd $VULKAN_VERSION
rm -rf x86_64

sed -i 's|sudo||g' vulkansdk
sed -i 's|apt-get install|apt-get install -y --no-install-recommends|g' vulkansdk

export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export CMAKE_POLICY_VERSION_MINIMUM="3.5"

printf "\nBuilding Vulkan SDK $VULKAN_VERSION\n"

./vulkansdk --maxjobs --no-deps

tarpack upload vulkan-sdk-$VULKAN_VERSION $(uname -m)/ || echo "failed to upload tarball"

cp -r $(uname -m)/* /usr/local/
cd $TMP

rm -rf vulkansdk* $VULKAN_VERSION
echo "installed" > "$TMP/.vulkan"

ldconfig