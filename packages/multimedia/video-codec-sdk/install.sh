#!/usr/bin/env bash
echo "Installing NVIDIA Video Codec SDK $NV_CODEC_VERSION (NVENC/CUVID)"
ZIP="Video_Codec_SDK_$NV_CODEC_VERSION.zip"
set -ex
cd $TMP

if [ ! -f $ZIP ]; then
  wget $WGET_FLAGS $MULTIARCH_URL/$ZIP
fi

unzip $ZIP
rm $ZIP

mkdir -p $SOURCE/build
mv Video_Codec_SDK_*/* $SOURCE/
cp $SOURCE/Lib/linux/stubs/$(uname -m)/*.so /usr/local/lib
cp $SOURCE/Interface/* /usr/local/include

cd $SOURCE/build
cmake ../Samples
make -j$(nproc)
make install 
