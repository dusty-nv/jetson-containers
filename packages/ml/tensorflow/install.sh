#!/usr/bin/env bash
set -ex

# organize CUDA how TF expects it
bash /tmp/TENSORFLOW/link_cuda.sh

# install LLVM
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
./llvm.sh 20 all

# TENSORFLOW C++ extensions frequently use ninja for parallel builds
pip3 install scikit-build ninja six numpy wheel

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of Tensorflow ${TENSORFLOW_VERSION}"
    exit 1
fi

# if TENSORFLOW_VERSION <= 2.16.1 download the wheel from the mirror
# if not install from the Jetson PyPI server ($PIP_INSTALL_URL)
if [ "$(echo "${TENSORFLOW_VERSION} <= 2.16.1" | bc 2>/dev/null || echo 0)" -eq 1 ]; then
    pip3 install 'setuptools==68.2.2'
    H5PY_SETUP_REQUIRES=0 pip3 install h5py
    pip3 install future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 futures pybind11
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${TENSORFLOW_URL} -O ${TENSORFLOW_WHL}
    pip3 install ${TENSORFLOW_WHL}
    rm ${TENSORFLOW_WHL}
else
    # install from the Jetson PyPI server ($PIP_INSTALL_URL)
    pip3 install tensorflow==${TENSORFLOW_VERSION}
fi
