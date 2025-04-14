#!/usr/bin/env bash
# JAX installer
set -ex

# bash /tmp/JAX/link_cuda.sh

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 20 all
ln -sf /usr/bin/llvm-config-* /usr/bin/llvm-config
ln -s /usr/bin/clang-1* /usr/bin/clang

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

# ensure JAX is installed correctly
python3 -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"CUDA devices: {jax.devices()}");'
