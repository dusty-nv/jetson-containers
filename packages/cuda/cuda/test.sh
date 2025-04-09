#!/usr/bin/env bash

echo "=== CUDA version files ==="
cat /usr/local/cuda/version*

echo ""
echo "=== Location of nvcc ==="
which nvcc

echo ""
echo "=== nvcc version ==="
nvcc --version

echo ""
echo "=== Supported GPU architectures by nvcc ==="
nvcc --list-gpu-arch
