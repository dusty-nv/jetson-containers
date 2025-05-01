#!/usr/bin/env bash
# JAX installer
set -ex

# bash /tmp/JAX/link_cuda.sh

# JAX C++ extensions frequently use ninja for parallel builds
pip3 install scikit-build ninja

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of JAX ${JAX_BUILD_VERSION}"
    exit 1
fi

# install from the Jetson PyPI server ($PIP_INSTALL_URL)
#pip3 install jax==${JAX_VERSION} jaxlib==${JAX_VERSION}
pip3 install jaxlib jax_cuda12_plugin opt_einsum
pip3 install --no-dependencies jax

if [ $(vercmp "$JAX_VERSION" "0.6.0") -ge 0 ]; then
    pip3 install 'ml_dtypes>=0.5' # missing float4_e2m1fn
fi

# ensure JAX is installed correctly
python3 -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"CUDA devices: {jax.devices()}");'
