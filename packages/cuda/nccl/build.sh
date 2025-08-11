#!/usr/bin/env bash
set -euxo pipefail

if [[ ! "$CUDA_ARCH" == "tegra-aarch64" ]]; then
    # The build stage is only for tegra-aarch64
    exit 1
fi

echo "Building NVIDIA NCCL $NCCL_VERSION (NCCL)"

apt-get update
apt-get install -y --no-install-recommends build-essential devscripts debhelper fakeroot

git clone --branch=v${NCCL_VERSION}-1 https://github.com/NVIDIA/nccl
cd nccl

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
