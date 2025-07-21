#!/usr/bin/env bash
# Python builder
set -ex

echo "Building PyTorch ${PYTORCH_BUILD_VERSION}"

# build from source
git clone --branch "v${PYTORCH_BUILD_VERSION}" --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch ||
git clone --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch
cd /opt/pytorch

# https://github.com/pytorch/pytorch/issues/138333
CPUINFO_PATCH=third_party/cpuinfo/src/arm/linux/aarch64-isa.c
sed -i 's|cpuinfo_log_error|cpuinfo_log_warning|' ${CPUINFO_PATCH}
grep 'PR_SVE_GET_VL' ${CPUINFO_PATCH} || echo "patched ${CPUINFO_PATCH}"
tail -20 ${CPUINFO_PATCH}

pip3 install -r requirements.txt
pip3 install scikit-build ninja
pip3 install 'cmake<4'

#TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
# https://github.com/pytorch/pytorch/pull/157791/files#diff-f271c3ed0c135590409465f4ad55c570c418d2c0509bbf1b1352ebdd1e6611d1
if [[ "$CUDA_VERSION" == "12.6" ]]; then
  export TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=balance"
else
  export TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=size"
fi

PYTORCH_BUILD_NUMBER=1 \
USE_CUDNN=1 \
USE_CUSPARSELT=1 \
USE_CUDSS=1 \
USE_CUFILE=1 \
USE_NATIVE_ARCH=1 \
USE_DISTRIBUTED=1 \
USE_FLASH_ATTENTION=1 \
USE_MEM_EFF_ATTENTION=1 \
USE_TENSORRT=0 \
USE_BLAS="$USE_BLAS" \
BLAS="$BLAS" \
python3 setup.py bdist_wheel --dist-dir /opt

cd /
rm -rf /opt/pytorch

# install the compiled wheel
pip3 install /opt/torch*.whl
python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'
twine upload --verbose /opt/torch*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

