#!/bin/bash

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
