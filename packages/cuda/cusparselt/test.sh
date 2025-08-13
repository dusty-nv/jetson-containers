#!/usr/bin/env bash
set -euo pipefail

nvcc /test/test_cusparselt.cu \
    -v \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -L/usr/lib/aarch64-linux-gnu \
    -lcusparseLt -lcusparse -lcudart \
    -Xlinker -rpath=/usr/local/cuda/lib64 -Xlinker -rpath=/usr/lib/aarch64-linux-gnu \
    -Wno-deprecated-gpu-targets \
    -Wno-deprecated-declarations \
    -o /test/test_cusparselt

/test/test_cusparselt