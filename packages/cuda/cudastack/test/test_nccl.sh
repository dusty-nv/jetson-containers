#!/usr/bin/env bash

echo "testing NCCL"

cat > /test/nccl_version_test.c << 'EOF'
#include <stdio.h>
#include <nccl.h>

int main() {
    int version;
    ncclResult_t result = ncclGetVersion(&version);
    if (result == ncclSuccess) {
        printf("NCCL Version: %d.%d.%d\n",
               (version / 10000),
               (version / 100) % 100,
               version % 100);
    } else {
        printf("Failed to get NCCL version: %s\n", ncclGetErrorString(result));
    }
    return 0;
}
EOF

echo "Compiling version test..."
if nvcc -o /test/nccl_version_test /test/nccl_version_test.c -lnccl 2>/dev/null; then
    /test/nccl_version_test
    rm -f /test/nccl_version_test /test/nccl_version_test.c
    echo "NCCL ok"
else
    echo "Failed to compile NCCL test - NCCL may not be properly installed!"
fi

if [[ ${ENABLE_DISTRIBUTED_JETSON_NCCL:-0} != "1" ]]; then
    echo "Skipping distributed NCCL test for Jetson, to enable build cudastack:distributed"
    exit 0
fi

echo "Testing distributed NCCL for Jetson ..."

git clone --depth=1 https://github.com/nvidia/nccl-tests

cd nccl-tests || exit

# Build only all_reduce_perf test
make -j -C src BUILDDIR=../build ../build/all_reduce_perf NVCC_GENCODE="-gencode=arch=compute_87,code=sm_87"

# Run on single node, scanning from 8 Bytes to 128MBytes :
if NCCL_NVLS_ENABLE=0 NCCL_DEBUG=TRACE ./build/all_reduce_perf -b 8 -e 128M 2>/dev/null; then
    echo '✅ distributed NCCL for Jetson is available'
else
    echo '❌ distributed NCCL for Jetson is NOT available'
fi

echo 'Cleaning up...'
cd ..
rm -rf nccl-tests
