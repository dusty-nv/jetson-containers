#!/usr/bin/env bash

cat /usr/include/aarch64-linux-gnu/cudnn_version* | grep CUDNN_M

CUDNN_SAMPLES=/usr/src/cudnn_samples_v8

if [ -d $CUDNN_SAMPLES ]; then
	cd $CUDNN_SAMPLES/conv_sample/
	./conv_sample
fi