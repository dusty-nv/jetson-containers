#!/usr/bin/env bash
#
# this script copies development files and headers from the target host
# into the packages dir, which get used during building some containers
#

mkdir -p packages/usr/include
mkdir -p packages/usr/include/aarch64-linux-gnu

cp /usr/include/cublas*.h packages/usr/include
cp /usr/include/cudnn*.h packages/usr/include

cp /usr/include/aarch64-linux-gnu/Nv*.h packages/usr/include/aarch64-linux-gnu


