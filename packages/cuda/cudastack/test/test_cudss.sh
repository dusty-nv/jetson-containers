#!/usr/bin/env bash
set -euo pipefail

echo "=== cuDSS CUDA Version Compatibility Test ==="

# 1. Check System CUDA Version
echo "1. System CUDA Version:"
CUDA_VERSION=$(nvcc --version | grep -Po 'V\K[0-9]+\.[0-9]+' || echo "unknown")
echo "CUDA Version detected: $CUDA_VERSION"

# 2. Locate libcudss.so
echo -e "\n2. Locating cuDSS library:"
CUDSS_LIB=$(find /usr /lib -name "libcudss.so.0" 2>/dev/null | head -n 1)

if [[ -z "$CUDSS_LIB" ]]; then
    echo "❌ libcudss.so.0 not found!"
    exit 1
fi
echo "Found at: $CUDSS_LIB"

# 3. Validate ABI Dependencies (The Critical Point)
echo -e "\n3. Checking ABI Dependencies (ldd):"
DEPENDENCIES=$(ldd "$CUDSS_LIB")
echo "$DEPENDENCIES"

if echo "$DEPENDENCIES" | grep -q "libcublas.so.13"; then
    echo "❌ CRITICAL ERROR: cuDSS is linked against libcublas.so.13 (CUDA 13)!"
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        echo "   This is a version mismatch for CUDA $CUDA_VERSION system."
        exit 1
    fi
elif echo "$DEPENDENCIES" | grep -q "libcublas.so.12"; then
    echo "✅ cuDSS is correctly linked against libcublas.so.12 (CUDA 12)."
else
    echo "⚠️  Could not explicitly determine libcublas version dependency via ldd."
fi

# 4. Test Linking with cuDSS
echo -e "\n4. Test Linking with cuDSS:"
cat > /tmp/test_cudss.cpp << 'EOF'
#include <cudss.h>
#include <iostream>
int main() {
    cudssHandle_t handle;
    cudssStatus_t status = cudssCreate(&handle);
    if (status == CUDSS_STATUS_SUCCESS) {
        std::cout << "✅ cuDSS Handle created successfully!" << std::endl;
        cudssDestroy(handle);
        return 0;
    } else {
        std::cerr << "❌ cuDSS Initialization failed: " << (int)status << std::endl;
        return 1;
    }
}
EOF

echo "Compiling test program..."
if g++ -o /tmp/test_cudss /tmp/test_cudss.cpp -lcudss -lcublas 2>&1; then
    echo "✅ Compilation and linking succeeded."
    echo "Running test execution..."
    if /tmp/test_cudss; then
        echo "✅ Execution succeeded."
    else
        echo "❌ Execution failed (Runtime Linker Error)."
        exit 1
    fi
else
    echo "❌ Compilation failed."
    exit 1
fi

echo -e "\n=== Test Complete: cuDSS is binary compatible ==="
