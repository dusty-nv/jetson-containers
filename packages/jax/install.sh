#!/usr/bin/env bash
# JAX installer
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

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of JAX ${JAX_BUILD_VERSION}"
    exit 1
fi

# install from the Jetson PyPI server ($PIP_INSTALL_URL)
pip3 install --verbose --no-cache-dir jax==${JAX_VERSION} jaxlib==${JAX_VERSION}

# ensure JAX is installed correctly
python3 -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"CUDA devices: {jax.devices()}");'

# JAX C++ extensions frequently use ninja for parallel builds
pip3 install --no-cache-dir scikit-build ninja