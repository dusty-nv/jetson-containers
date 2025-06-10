#!/usr/bin/env bash
if [ "$BUILD_SAMPLES" != "on" ]; then
  exit 1; # skip building samples
fi

echo "Building NVIDIA Video Codec SDK $NV_CODEC_VERSION (NVENC/CUVID)"
set -ex

cd $SOURCE/build
cmake ../Samples
make -j$(nproc)
make install 
