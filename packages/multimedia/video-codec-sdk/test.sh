#!/usr/bin/env bash
set -ex

# Check ffnvcodec headers (installed by nv-codec-headers make install)
ls /usr/local/include/ffnvcodec/nvEncodeAPI.h
ls /usr/local/include/ffnvcodec/dynlink_nvcuvid.h
ls /usr/local/include/ffnvcodec/dynlink_cuviddec.h

# Check pkg-config
pkg-config --modversion ffnvcodec

# Check that libnvcuvid and libnvidia-encode are linkable (stubs, driver, or system libs)
ldconfig -p | grep -q libnvcuvid || ls /usr/local/lib/libnvcuvid* || ls ${CUDA_HOME:-/usr/local/cuda}/lib64/stubs/libnvcuvid*
ldconfig -p | grep -q libnvidia-encode || ls /usr/local/lib/libnvidia-encode* || ls ${CUDA_HOME:-/usr/local/cuda}/lib64/stubs/libnvidia-encode*
