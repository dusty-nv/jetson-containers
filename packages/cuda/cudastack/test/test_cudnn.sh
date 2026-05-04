#!/usr/bin/env bash
set -e

echo "=== cuDNN headers ==="
CUDNN_VERSION_HEADER=""

for header in \
    /usr/include/*-linux-gnu/cudnn_version*.h \
    /usr/include/cudnn_version*.h
do
    if [ -f "${header}" ]; then
        CUDNN_VERSION_HEADER="${header}"
        break
    fi
done

if [ -z "${CUDNN_VERSION_HEADER}" ]; then
    echo "cuDNN version header not found"
    exit 1
fi

echo "Using ${CUDNN_VERSION_HEADER}"
grep CUDNN_M "${CUDNN_VERSION_HEADER}"

echo ""
echo "=== cuDNN library ==="
ldconfig -p | grep libcudnn || true

cat > /tmp/test_cudnn_version.c <<'EOF'
#include <stdio.h>
#include <cudnn.h>

int main(void) {
    printf("cuDNN runtime version: %zu\n", cudnnGetVersion());
    return 0;
}
EOF

echo "Compiling cuDNN version test..."
gcc /tmp/test_cudnn_version.c -o /tmp/test_cudnn_version \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lcudnn
/tmp/test_cudnn_version
rm -f /tmp/test_cudnn_version /tmp/test_cudnn_version.c

CUDNN_SAMPLE_DIR="$(find /usr/src -maxdepth 3 -type d -path '*/conv_sample' 2>/dev/null | head -n 1)"

if [ -n "${CUDNN_SAMPLE_DIR}" ] && [ -f "${CUDNN_SAMPLE_DIR}/Makefile" ]; then
    echo ""
    echo "=== cuDNN conv_sample ==="
    cd "${CUDNN_SAMPLE_DIR}"
    if [ ! -f conv_sample ]; then
        echo "building cuDNN conv_sample"
        make -j"$(nproc)"
    fi
    ./conv_sample
else
    echo "cuDNN conv_sample not found with a Makefile - skipping sample test"
fi

echo "cuDNN ok"