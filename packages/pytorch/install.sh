#!/usr/bin/env bash
set -ex

# install prerequisites
apt-get update
apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libomp-dev
        
if [ $USE_MPI == 1 ]; then
  apt-get install -y --no-install-recommends \
          libopenmpi-dev \
          openmpi-bin \
          openmpi-common \
          gfortran
fi

rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of PyTorch ${PYTORCH_BUILD_VERSION}"
	exit 1
fi

# on x86_64, install from pytorch nightly server
# on aarch64, install from the Jetson pypi server ($PIP_INSTALL_URL)
pip3 install torch==${TORCH_VERSION} || \
pip3 install --pre "torch>=${PYTORCH_BUILD_VERSION}.dev,<=${PYTORCH_BUILD_VERSION}"

# reinstall numpy<2 on CUDA < 12.8
bash /tmp/numpy/install.sh

# make sure it loads
python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'

# PyTorch C++ extensions frequently use ninja parallel builds
pip3 install scikit-build ninja
   
# temporary patches to enable newer blackwell sm's
if [[ "$(uname -m)" == "x86_64" && ${TORCH_VERSION} == "2.6" ]]; then
  PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2`
  TORCH_PATCH="$PYTHON_ROOT/torch/utils/cpp_extension.py"
  echo "patching ${TORCH_PATCH}"
  sed -i "s|'10.0']|'10.0', '10.0a', '10.1', '10.1a', '12.0', '12.0a']|" ${TORCH_PATCH}
  cat ${TORCH_PATCH} | grep '10.0'
fi