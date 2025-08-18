#!/usr/bin/env bash
set -ex
echo "Building torchaudio ${TORCHAUDIO_VERSION}"

# --- Install required system dependencies ---
apt-get update
apt-get install -y --no-install-recommends \
    git \
    pkg-config \
    libffi-dev \
    libsndfile1

rm -rf /var/lib/apt/lists/*
apt-get clean

# --- Clone torchaudio repository (try versioned tags first, fallback to release branch, then master) ---
git clone --branch=v${TORCHAUDIO_VERSION} --recursive --depth=1 https://github.com/pytorch/audio /opt/torchaudio \
  || git clone --branch=release/${BRANCH_VERSION} --recursive --depth=1 https://github.com/pytorch/audio /opt/torchaudio \
  || git clone --recursive --depth=1 https://github.com/pytorch/audio /opt/torchaudio

cd /opt/torchaudio

# --- https://github.com/pytorch/audio/pull/3811
sed -i '1i#include <float.h>' src/libtorchaudio/cuctc/src/ctc_prefix_decoder_kernel_v2.cu || echo "warning:  failed to patch ctc_prefix_decoder_kernel_v2.cu"

# --- Patch forced_align/gpu/compute.cu: replace cub::Max -> thrust::maximum ---
FA_FILE="src/libtorchaudio/forced_align/gpu/compute.cu"

# Ensure required includes (CUB for BlockReduce, Thrust for maximum/minimum functors)
grep -q 'cub/block/block_reduce.cuh' "$FA_FILE" || sed -i '1i #include <cub/block/block_reduce.cuh>' "$FA_FILE"
grep -q '<thrust/functional.h>'      "$FA_FILE" || sed -i '1i #include <thrust/functional.h>'      "$FA_FILE"

# Replace deprecated CUB functors with Thrust equivalents
sed -i 's/cub::Max()/thrust::maximum<scalar_t>()/g' "$FA_FILE"
sed -i 's/cub::Min()/thrust::minimum<scalar_t>()/g' "$FA_FILE"
# --- end of patch ---

# --- CUDA / CUB patches (for CUDA 13 compatibility) ---
# 1) Patch ctc_prefix_decoder_kernel_v2.cu: add missing includes and replace FpLimits
CTC_FILE="src/libtorchaudio/cuctc/src/ctc_prefix_decoder_kernel_v2.cu"
if [ -f "$CTC_FILE" ]; then
  # Add includes if not already present
  grep -q 'cub/cub.cuh' "$CTC_FILE" || sed -i '1i #include <cub/cub.cuh>' "$CTC_FILE"
  grep -q '<limits>'     "$CTC_FILE" || sed -i '1i #include <limits>'     "$CTC_FILE"

  # Replace FpLimits -> std::numeric_limits::lowest
  sed -i 's/cub::FpLimits<float>::Lowest()/std::numeric_limits<float>::lowest()/g' "$CTC_FILE"
fi

# 2) Ensure compute.cu also includes CUB to make cub::Max available
if [ -f "$FA_FILE" ]; then
  grep -q 'cub/cub.cuh' "$FA_FILE" || sed -i '1i #include <cub/cub.cuh>' "$FA_FILE"
fi
# --- End of CUDA patches ---

# (Optional) Legacy patch you might have already had
sed -i '1i#include <float.h>' "$CTC_FILE" || echo "warning: failed to patch <$CTC_FILE> with <float.h>"

# --- Build wheel ---
BUILD_VERSION=${TORCHAUDIO_VERSION} \
BUILD_SOX=1 \
python3 setup.py bdist_wheel --verbose --dist-dir /opt

cd ../
rm -rf /opt/torchaudio

# --- Install and verify ---
pip3 install /opt/torchaudio*.whl
pip3 show torchaudio && python3 -c 'import torchaudio; print(torchaudio.__version__);'

# --- Upload (if configured) ---
twine upload --verbose /opt/torchaudio*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
