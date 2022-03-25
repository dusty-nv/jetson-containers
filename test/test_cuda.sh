#!/usr/bin/env bash

echo "testing CUDA..."

nvcc --version

cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery

echo "CUDA OK"