#!/usr/bin/env bash

nvcc /test/test_nvshmem.cu \
  -I/usr/include/nvshmem \
  -I/usr/local/cuda/include \
  -L/usr/lib/aarch64-linux-gnu \
  -lnvshmem_host -lnvshmem -lcudart \
  -Xlinker -rpath=/usr/lib/aarch64-linux-gnu \
  -Wno-deprecated-gpu-targets \
  -Wno-deprecated-declarations \
  -o /test/test_nvshmem

/test/test_nvshmem
