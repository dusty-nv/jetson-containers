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

PYTORCH_BUILD_NUMBER=ON \
USE_CUDNN=ON \
USE_CUSPARSELT=ON \
USE_CUDSS=ON \
USE_CUFILE=ON \
USE_NATIVE_ARCH=ON \
USE_DISTRIBUTED=ON \
USE_FLASH_ATTENTION=ON \
USE_MEM_EFF_ATTENTION=ON \
USE_TENSORRT=OFF \
python3 setup.py bdist_wheel --dist-dir /opt

cd /
rm -rf /opt/pytorch

# install the compiled wheel
pip3 install /opt/torch*.whl
python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'
twine upload --verbose /opt/torch*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
