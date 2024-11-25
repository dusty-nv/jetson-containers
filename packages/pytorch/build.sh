#!/usr/bin/env bash
# Python builder
set -ex

echo "Building PyTorch ${PYTORCH_BUILD_VERSION}"
   
# build from source
git clone --branch "v${PYTORCH_BUILD_VERSION}" --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch
cd /opt/pytorch

# https://github.com/pytorch/pytorch/issues/138333
CPUINFO_PATCH=third_party/cpuinfo/src/arm/linux/aarch64-isa.c
sed -i 's|cpuinfo_log_error|cpuinfo_log_warning|' ${CPUINFO_PATCH}
grep 'PR_SVE_GET_VL' ${CPUINFO_PATCH}
tail -20 ${CPUINFO_PATCH}

pip3 install --no-cache-dir -r requirements.txt
pip3 install --no-cache-dir scikit-build ninja

PYTORCH_BUILD_NUMBER=1 \
TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
USE_NATIVE_ARCH=1 \
USE_DISTRIBUTED=1 \
USE_TENSORRT=0 \
USE_FBGEMM=0 \
python3 setup.py bdist_wheel --dist-dir /opt

cd /
rm -rf /opt/pytorch
mkdir -p /torch-wheels
cp /opt/torch*.whl /torch-wheels/

# install the compiled wheel
pip3 install /opt/torch*.whl
python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'
twine upload --verbose /opt/torch*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
