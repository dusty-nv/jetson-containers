#!/usr/bin/env bash

cat /usr/include/aarch64-linux-gnu/cudnn_version* | grep CUDNN_M

CUDNN_SAMPLES=/usr/src/cudnn_samples_v*

if [ -d $CUDNN_SAMPLES ]; then
	cd $CUDNN_SAMPLES/conv_sample/
	if [ ! -f conv_sample ]; then
		echo "building cuDNN conv_sample"
		make -j$(nproc)
	fi
	./conv_sample
fi