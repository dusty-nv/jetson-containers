#!/usr/bin/env bash
set -ex

: "${CUDA_SAMPLES_ROOT:=/opt/cuda-samples}"

cd $CUDA_SAMPLES_ROOT/bin/$(uname -m)/linux/release

./deviceQuery
./bandwidthTest
./vectorAdd
./matrixMul