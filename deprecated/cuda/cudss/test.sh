
#!/usr/bin/env bash
set -euo pipefail

echo "=== cuDSS CUDA Version Compatibility Test ==="

# Check what CUDA version is actually installed
echo "1. System CUDA Version:"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | grep -E "V[0-9]+\.[0-9]+"
else
    echo "nvcc not found"
fi

# Check CUDA library versions
echo -e "\n2. CUDA Library Versions:"
if [[ -d "/usr/local/cuda/lib64" ]]; then
    echo "CUDA 13.x libraries:"
    ls -la /usr/local/cuda/lib64/libcublas* | head -5
fi

# Check what cuDSS library expects
echo -e "\n3. cuDSS Library Dependencies:"
if [[ -f "/lib/aarch64-linux-gnu/libcudss.so.0" ]]; then
    echo "cuDSS library found:"
    ls -la /lib/aarch64-linux-gnu/libcudss.so.0

    echo -e "\ncuDSS library dependencies:"
    ldd /lib/aarch64-linux-gnu/libcudss.so.0 | grep cublas || echo "No cublas dependency found"

    echo -e "\ncuDSS library symbols (cublas functions):"
    nm -D /lib/aarch64-linux-gnu/libcudss.so.0 | grep cublas | head -5 || echo "No cublas symbols found"
else
    echo "cuDSS library not found at /lib/aarch64-linux-gnu/libcudss.so.0"
fi

# Test linking with cuDSS
echo -e "\n4. Test Linking with cuDSS:"
cat > /tmp/test_cudss.cpp << 'EOF'
#include <iostream>
int main() {
    std::cout << "Testing cuDSS linking..." << std::endl;
    return 0;
}
EOF

echo "Compiling test program with cuDSS..."
if g++ -o /tmp/test_cudss /tmp/test_cudss.cpp -lcudss 2>&1; then
    echo "✅ Linking with cuDSS succeeded"
else
    echo "❌ Linking with cuDSS failed - this confirms the version mismatch"
fi

# Check if we can find CUDA 12.x libraries anywhere
echo -e "\n5. Searching for CUDA 12.x Libraries:"
find /usr -name "libcublas.so.12*" 2>/dev/null | head -5 || echo "No CUDA 12.x libraries found"

# Check package manager for cuDSS
echo -e "\n6. cuDSS Package Information:"
if command -v dpkg >/dev/null 2>&1; then
    dpkg -l | grep -i cudss || echo "No cuDSS packages found via dpkg"
fi

echo -e "\n=== Test Complete ==="