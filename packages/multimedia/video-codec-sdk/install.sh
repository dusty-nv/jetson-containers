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

# Stubs from CUDA toolkit
for lib in libnvcuvid.so libnvidia-encode.so; do
    if [ -f "${CUDA_HOME:-/usr/local/cuda}/lib64/stubs/$lib" ]; then
        cp "${CUDA_HOME:-/usr/local/cuda}/lib64/stubs/$lib" /usr/local/lib/
    fi
done
ldconfig

cd /tmp && rm -rf nv-codec-headers

echo "NVIDIA Video Codec SDK ${NV_CODEC_VERSION} installed successfully"
