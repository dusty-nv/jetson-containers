#!/usr/bin/env bash
set -ex

bash /tmp/TENSORFLOW/link_cuda.sh

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 17 all
ln -sf /usr/bin/llvm-config-* /usr/bin/llvm-config
ln -s /usr/bin/clang-1* /usr/bin/clang

# TENSORFLOW C++ extensions frequently use ninja for parallel builds
pip3 install --no-cache-dir scikit-build ninja

# install prerequisites - https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
apt-get update
apt-get install -y --no-install-recommends \
        liblapack-dev \
        libblas-dev \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev \
        zlib1g-dev \
        libjpeg8-dev \
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        libncurses5-dev \
        libnss3-dev \
        libreadline-dev \
        libsqlite3-dev \
        patchelf

rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of Tensorflow ${TENSORFLOW_VERSION}"
    exit 1
fi

# if TENSORFLOW_VERSION <= 2.16.1 download the wheel from the mirror if not # install from the Jetson PyPI server ($PIP_INSTALL_URL)
if [ $(echo "${TENSORFLOW_VERSION} <= 2.16.1" | bc) -eq 1 ]; then
    pip3 install --no-cache-dir 'setuptools==68.2.2'
    H5PY_SETUP_REQUIRES=0 pip3 install --no-cache-dir --verbose h5py
    pip3 install --no-cache-dir --verbose future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 futures pybind11
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${TENSORFLOW_URL} -O ${TENSORFLOW_WHL}
    pip3 install --no-cache-dir --verbose ${TENSORFLOW_WHL}
    rm ${TENSORFLOW_WHL}
else
    # install from the Jetson PyPI server ($PIP_INSTALL_URL)
    pip3 install --no-cache-dir --verbose ${TENSORFLOW_VERSION}
fi

# Verify the installation
python3 -c "import tensorflow as tf; print(tf.__version__)"

