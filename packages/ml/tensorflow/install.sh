#!/usr/bin/env bash
set -ex

# organize CUDA how TF expects it
bash /tmp/TENSORFLOW/link_cuda.sh

# TENSORFLOW C++ extensions frequently use ninja for parallel builds
uv pip install scikit-build ninja six numpy wheel

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of Tensorflow ${TENSORFLOW_VERSION}"
    exit 1
fi

# if TENSORFLOW_VERSION <= 2.16.1 download the wheel from the mirror
# if not install from the Jetson PyPI server ($PIP_INSTALL_URL)
if [ "$(echo "${TENSORFLOW_VERSION} <= 2.16.1" | bc 2>/dev/null || echo 0)" -eq 1 ]; then
    uv pip install 'setuptools==68.2.2'
    H5PY_SETUP_REQUIRES=0 uv pip install h5py
    uv pip install future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 futures
    wget $WGET_FLAGS ${TENSORFLOW_URL} -O ${TENSORFLOW_WHL}
    uv pip install ${TENSORFLOW_WHL}
    rm ${TENSORFLOW_WHL}
else
    # install from the Jetson PyPI server ($PIP_INSTALL_URL)
    uv pip install tensorflow==${TENSORFLOW_VERSION}
fi
