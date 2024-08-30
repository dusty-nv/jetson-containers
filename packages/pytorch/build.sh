#!/usr/bin/env bash
# Python builder
set -ex

echo "Building PyTorch ${PYTORCH_BUILD_VERSION}"
   
# install prerequisites
apt-get update
apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        gfortran \
        libomp-dev
rm -rf /var/lib/apt/lists/*
apt-get clean

# build from source
git clone --branch "v${PYTORCH_BUILD_VERSION}" --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch
cd /opt/pytorch

pip3 install --no-cache-dir -r requirements.txt
pip3 install --no-cache-dir scikit-build ninja

PYTORCH_BUILD_NUMBER=1 \
TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
USE_QNNPACK=0 \
USE_PYTORCH_QNNPACK=0 \
USE_NATIVE_ARCH=1 \
USE_DISTRIBUTED=1 \
USE_TENSORRT=0 \
python3 setup.py bdist_wheel --dist-dir /opt

cd /
rm -rf /opt/pytorch
cp /opt/torch*.whl /torch-wheels

# install the compiled wheel
pip3 install /opt/torch*.whl
python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'
twine upload --verbose /opt/torch*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
