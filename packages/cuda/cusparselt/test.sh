#!/usr/bin/env bash
set -euo pipefail

rm -f /test/test_cusparselt || true

nvcc /test/test_cusparselt.cu \
    -v \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -L/usr/lib/aarch64-linux-gnu \
    -lcusparseLt -lcusparse -lcudart \
    -Xlinker -rpath=/usr/local/cuda/lib64 -Xlinker -rpath=/usr/lib/aarch64-linux-gnu \
    -Wno-deprecated-gpu-targets \
    -Wno-deprecated-declarations \
    -o /test/test_cusparselt

# Show supported SM architectures in the library
echo "🔍 === cusparseLt library supported architectures ==="
ARCHS=$(cuobjdump --dump-sass /usr/lib/aarch64-linux-gnu/libcusparseLt.so.0.7.1.0 2>/dev/null | grep -E "^arch = sm_" | sort -u || true)
if [ -n "$ARCHS" ]; then
    echo "$ARCHS"
    echo ""
    echo "📊 Summary: Library supports $(echo "$ARCHS" | wc -l) SM architectures"

    # Check specifically for Thor support
    if echo "$ARCHS" | grep -q "sm_110"; then
        echo "✅ Thor (sm_110) support: YES"
    else
        echo "❌ Thor (sm_110) support: NO"
    fi
else
    echo "❌ Could not determine supported architectures"
fi

echo ""
echo "🚀 === Running cusparseLt test ==="
/test/test_cusparselt