#!/usr/bin/env bash
# JAX installer
set -ex

# bash /tmp/JAX/link_cuda.sh
apt-get update && \
apt-get install -y --no-install-recommends \
    vim-common \
    xxd \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean

# JAX C++ extensions frequently use ninja for parallel builds
uv pip install scikit-build ninja

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of JAX ${JAX_BUILD_VERSION}"
    exit 1
fi

# install from the Jetson PyPI server ($PIP_INSTALL_URL)
uv pip install jaxlib==${JAX_VERSION} jax_cuda13_plugin opt_einsum
uv pip install --no-deps jax==${JAX_VERSION}

if [ $(vercmp "$JAX_VERSION" "0.6.0") -ge 0 ]; then
    uv pip install 'ml_dtypes>=0.5' # missing float4_e2m1fn
fi

# ensure JAX is installed correctly
python3 -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"CUDA devices: {jax.devices()}");'
