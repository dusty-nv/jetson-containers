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
cp $SOURCE/Lib/linux/stubs/$(uname -m)/*.so $CUDA_HOME/lib64

cp $SOURCE/Interface/* /usr/local/include
cp $SOURCE/Interface/* $CUDA_HOME/include

