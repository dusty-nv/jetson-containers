#!/bin/bash
set -eux

# The build stage is only for tegra-aarch64
if [ "$CUDA_ARCH" != "tegra-aarch64" ]; then
    echo "build_nccl.sh is only for tegra-aarch64, skipping (current: $CUDA_ARCH)"
    exit 1
fi

echo "Building NVIDIA NCCL $NCCL_VERSION (NCCL)"

apt-get update
apt-get install -y --no-install-recommends build-essential devscripts debhelper fakeroot

git clone --branch=v${NCCL_VERSION}-1 https://github.com/NVIDIA/nccl
cd nccl

# Experimental support for distributed GPU communication on Jetson
if [[ "${ENABLE_DISTRIBUTED_JETSON_NCCL:-0}" == "1" && -f "${TMP}/cuda-stack/build/nccl-${NCCL_VERSION}.diff" ]]; then
	echo "Applying patch for distributed NCCL (for Jetson)"
	if ! git apply --check "${TMP}/cuda-stack/build/nccl-${NCCL_VERSION}.diff"; then
		echo "Patch for distributed NCCL (for Jetson) does not apply cleanly, skipping!"
	else
		git apply "${TMP}/cuda-stack/build/nccl-${NCCL_VERSION}.diff"
	fi
else
	echo "Skipping distributed NCCL (for Jetson) patch."
fi

make -j src.build NVCC_GENCODE="-gencode=arch=compute_87,code=sm_87"
make pkg.txz.build NVCC_GENCODE="-gencode=arch=compute_87,code=sm_87"

mkdir -p build/pkg/txz/lib

for f in build/pkg/txz/*.txz; do
  # extract to lib dir to upload with tarpack
  tar -xvJf "$f" -C build/pkg/txz/lib --strip-components=1
  # install on current host
  tar -xvJf "$f" -C /usr/local/ --strip-components=1
done

ls -ld build/pkg/txz/lib
ldconfig -p | grep nccl || true

tarpack upload nccl-${NCCL_VERSION} build/pkg/txz/lib || echo "failed to upload tarball"

cd ..

rm -rf build/pkg/txz
