#!/usr/bin/env bash
# PyTorch installer
set -ex

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

# install from the Jetson pypi server ($PIP_INSTALL_URL)
pip3 install --verbose --no-cache-dir torch==${TORCH_VERSION}

# make sure it loads
python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'

# patch for https://github.com/pytorch/pytorch/issues/45323
PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2`
TORCH_CMAKE_CONFIG="$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake"
echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}"
sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}

# PyTorch C++ extensions frequently use ninja parallel builds
pip3 install --no-cache-dir scikit-build ninja
   