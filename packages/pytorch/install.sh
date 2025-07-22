#!/usr/bin/env bash
set -ex

echo "##### 🏢 \$PYTORCH_OFFICIAL_WHL is $PYTORCH_OFFICIAL_WHL #####"

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

UBUNTU_VERSION=$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2)

if [[ "$UBUNTU_VERSION" == "22.04" ]]; then
  echo "🟢 Ubuntu 22.04 detected — installing GCC 13 for GLIBCXX_3.4.32"

  apt-get update
  apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    ca-certificates

  add-apt-repository ppa:ubuntu-toolchain-r/test -y
  apt-get update
  apt install -y g++-13 libstdc++-13-dev libstdc++6

  STDCPP_SO=$(find /usr/lib/aarch64-linux-gnu -name 'libstdc++.so.6.0.*' | sort -V | tail -n1)
  ln -sf "$STDCPP_SO" /usr/lib/aarch64-linux-gnu/libstdc++.so.6
  export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
  strings "$STDCPP_SO" | grep GLIBCXX_3.4.32

else
  echo "🔶 Ubuntu version is $UBUNTU_VERSION — skipping GCC upgrade"
fi

rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of PyTorch ${PYTORCH_BUILD_VERSION}"
	exit 1
fi

if [ "$PYTORCH_OFFICIAL_WHL" == "on" ]; then
	echo "##### 🏢Using official PyTorch 2.8 WHL #####"
  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
else
  # on x86_64, install from pytorch nightly server
  # on aarch64, install from the Jetson pypi server ($PIP_INSTALL_URL)
  pip3 install torch==${TORCH_VERSION} || \
  pip3 install --pre "torch>=${PYTORCH_BUILD_VERSION}.dev,<=${PYTORCH_BUILD_VERSION}"
fi
# reinstall numpy<2 on CUDA < 12.9
bash /tmp/numpy/install.sh

# make sure it loads
python3 -c 'import torch; \
    print(f"PyTorch version: {torch.__version__}"); \
    print(f"CUDA device #  : {torch.cuda.device_count()}"); \
    print(f"CUDA version   : {torch.version.cuda}"); \
    print(f"cuDNN version  : {torch.backends.cudnn.version()}");'
# PyTorch C++ extensions frequently use ninja parallel builds
pip3 install scikit-build ninja
