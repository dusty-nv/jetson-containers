#!/usr/bin/env bash
set -ex

: "${CUDA_SAMPLES_ROOT:=/opt/cuda-samples}"

function make_dirs() {
  cd $CUDA_SAMPLES_ROOT/Samples/1_Utilities/deviceQuery 
  make
  cd ../bandwidthTest
  make
  cd ../0_Introduction/matrixMul
  make
  cd ../vectorAdd
  make
}

function make_flat() {
  cd $CUDA_SAMPLES_ROOT/Samples/deviceQuery 
  make
  cd ../bandwidthTest
  make
  cd ../matrixMul
  make
  cd ../vectorAdd
  make
}

function make_all() {
  cd $CUDA_SAMPLES_ROOT
  make -j$(nproc) || echo "failed to make all CUDA samples"
}

function cmake_all() {
  apt-get update
  apt-get install -y --no-install-recommends \
        libfreeimage-dev
  rm -rf /var/lib/apt/lists/*
  apt-get clean

  cd $CUDA_SAMPLES_ROOT

  if [ $(uname -m) == "aarch64" ]; then
    local patch="Samples/3_CUDA_Features/CMakeLists.txt"
    sed -i 's|add_subdirectory(cdp.*|#|g' $patch
    echo "Patched $patch"
    cat $patch
  fi

  mkdir build
  cd build
  cmake ../
  make -j$(nproc) || echo "failed to cmake all CUDA samples"
  make install -j$(nproc)

  local out="$CUDA_SAMPLES_ROOT/bin/$(uname -m)/linux/release"
  mkdir -p $out || true;
  set +x

  for i in $(find ./Samples -type d); do
    local exe="$i/$(basename $i)"
    if [ -f "$exe" ]; then
      echo "Installing $exe -> $out"
      cp $exe $out
    fi
  done

  rm -rf $CUDA_SAMPLES_ROOT/build
}

if [ "$CUDA_SAMPLES_MAKE" == "make" ]; then
  make_dirs
  make_all
elif [ "$CUDA_SAMPLES_MAKE" == "make_flat" ]; then
  make_flat
  make_all
else
  cmake_all
fi
