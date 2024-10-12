#!/usr/bin/env bash
# JAX installer
set -ex

# bash /tmp/JAX/link_cuda.sh

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 18
ln -sf /usr/bin/llvm-config-* /usr/bin/llvm-config
ln -s /usr/bin/clang-1* /usr/bin/clang

# JAX C++ extensions frequently use ninja for parallel builds
pip3 install --no-cache-dir scikit-build ninja

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of JAX ${JAX_BUILD_VERSION}"
    exit 1
fi
# install from the Jetson PyPI server ($PIP_INSTALL_URL)
#pip3 install --verbose --no-cache-dir jax==${JAX_VERSION} jaxlib==${JAX_VERSION}
pip3 install --verbose --no-cache-dir jaxlib jax_cuda12_plugin opt_einsum
pip3 install --verbose --no-cache-dir --no-dependencies jax

# ensure JAX is installed correctly
python3 -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"CUDA devices: {jax.devices()}");'
