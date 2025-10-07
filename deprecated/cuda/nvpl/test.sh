#!/usr/bin/env bash
set -e

echo "=== Testing NVPL Installation ==="

# Check if NVPL package is installed
echo "Checking NVPL package installation..."
dpkg -l | grep -i nvpl || echo "No NVPL packages found"

# Check NVPL repository
echo "Checking NVPL repository..."
ls -la /var/nvpl-local-repo-* 2>/dev/null || echo "No NVPL repository found"

# Check for NVPL libraries
echo "Checking for NVPL libraries..."
find /usr -name "*nvpl*" -type f 2>/dev/null | head -20 || echo "No NVPL files found in /usr"

# Check for the specific BLAS library that PyTorch needs
echo "Checking for libnvpl_lapack_lp64_gomp.so.0..."
find /usr -name "libnvpl_lapack_lp64_gomp.so*" 2>/dev/null || echo "libnvpl_lapack_lp64_gomp.so not found"

# Check library paths
echo "Checking library paths..."
ldconfig -p | grep -i nvpl || echo "No NVPL libraries in ldconfig cache"

# Check if libraries are loadable
echo "Checking if NVPL libraries are loadable..."
find /usr -name "libnvpl*.so*" 2>/dev/null | while read lib; do
    echo "Testing library: $lib"
    if ldd "$lib" >/dev/null 2>&1; then
        echo "  ✓ Library is loadable"
    else
        echo "  ✗ Library failed to load"
    fi
done

# Check for NVPL headers
echo "Checking for NVPL headers..."
find /usr -name "*nvpl*" -type f -name "*.h" 2>/dev/null | head -10 || echo "No NVPL headers found"

# Check NVPL version
echo "Checking NVPL version..."
nvpl_version=$(dpkg -l | grep nvpl | head -1 | awk '{print $3}' || echo "unknown")
echo "NVPL version: $nvpl_version"

# Test basic NVPL functionality if available
echo "Testing NVPL basic functionality..."
if command -v nvpl-config >/dev/null 2>&1; then
    echo "nvpl-config found:"
    nvpl-config --version || echo "nvpl-config --version failed"
else
    echo "nvpl-config not found"
fi

# Check environment variables
echo "Checking NVPL environment variables..."
env | grep -i nvpl || echo "No NVPL environment variables found"

# Check if the specific library path is in LD_LIBRARY_PATH
echo "Checking LD_LIBRARY_PATH..."
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"

# Try to find the library in standard locations
echo "Searching for libnvpl_lapack_lp64_gomp.so.0 in standard locations..."
for path in /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/nvpl/lib; do
    if [ -d "$path" ]; then
        echo "Checking $path:"
        find "$path" -name "libnvpl_lapack_lp64_gomp.so*" 2>/dev/null || echo "  Not found"
    fi
done

echo "=== NVPL Test Complete ==="