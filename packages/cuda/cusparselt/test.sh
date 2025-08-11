#!/usr/bin/env bash

nvcc /test/test_cusparselt.cu \
    -I/usr/local/cuda/include \
    -L/usr/lib/aarch64-linux-gnu \
    -lcusparseLt -lcudart \
    -Xlinker -rpath=/usr/lib/aarch64-linux-gnu \
    -Wno-deprecated-gpu-targets \
    -Wno-deprecated-declarations \
    -o /test/test_cusparselt

/test/test_cusparselt
