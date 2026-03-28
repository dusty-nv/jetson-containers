#!/usr/bin/env bash
# Install NVIDIA Video Codec SDK headers from nv-codec-headers (public, no login)
# https://github.com/FFmpeg/nv-codec-headers
echo "Installing NVIDIA Video Codec SDK ${NV_CODEC_VERSION} (NVENC/CUVID)"
set -ex

cd /tmp

git clone --branch ${NV_CODEC_HEADERS_TAG} --depth 1 \
    https://github.com/FFmpeg/nv-codec-headers.git

cd nv-codec-headers

# Standard install: headers to /usr/local/include/ffnvcodec/ + pkg-config
make install PREFIX=/usr/local

# Also install to CUDA_HOME for projects that look there
if [ -n "${CUDA_HOME:-}" ]; then
    make install PREFIX=$CUDA_HOME
fi

# Copy individual headers to /usr/local/include/ for direct #include compatibility
cp include/ffnvcodec/nvEncodeAPI.h /usr/local/include/
for hdr in dynlink_nvcuvid.h dynlink_cuviddec.h; do
    src="include/ffnvcodec/${hdr}"
    dst="/usr/local/include/$(echo $hdr | sed 's/dynlink_//')"
    [ -f "$src" ] && cp "$src" "$dst"
done

# Copy or create stubs for build-time linking
CUDA_STUBS="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs"
SEARCH_PATHS="$CUDA_STUBS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib"

for lib in libnvcuvid.so libnvidia-encode.so; do
    found=""
    for dir in $SEARCH_PATHS; do
        if [ -f "$dir/$lib" ]; then
            cp "$dir/$lib" /usr/local/lib/
            found=1
            break
        fi
    done
    if [ -z "$found" ]; then
        echo "Creating minimal stub for $lib (not found in toolkit or driver paths)"
        echo "void __stub_$(echo $lib | tr '.-' '_')(){}" | \
            gcc -shared -o /usr/local/lib/$lib -x c - 2>/dev/null || true
    fi
done
ldconfig

cd /tmp && rm -rf nv-codec-headers

echo "NVIDIA Video Codec SDK ${NV_CODEC_VERSION} installed successfully"
