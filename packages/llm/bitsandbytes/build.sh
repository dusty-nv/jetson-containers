#!/usr/bin/env bash
set -ex

echo " ================ Building bitsandbytes ${BITSANDBYTES_VERSION} ================"

echo "### CUDA_INSTALLED_VERSION: $CUDA_INSTALLED_VERSION"
echo "### CUDA_MAKE_LIB: $CUDA_MAKE_LIB"
uv pip uninstall bitsandbytes || echo "previous bitsandbytes installation not found"

git clone --branch=$BITSANDBYTES_BRANCH --recursive --depth=1 "https://github.com/$BITSANDBYTES_REPO" /opt/bitsandbytes || \
git clone --recursive --depth=1 "https://github.com/$BITSANDBYTES_REPO" /opt/bitsandbytes
cd /opt/bitsandbytes
build_capability=${CUDAARCHS}
if [ $CUDA_INSTALLED_VERSION < 126 ]; then
    CUDA_VERSION=$CUDA_INSTALLED_VERSION make -C /opt/bitsandbytes -j$(nproc) "${CUDA_MAKE_LIB}"
    CUDA_VERSION=$CUDA_INSTALLED_VERSION make -C /opt/bitsandbytes -j$(nproc) "${CUDA_MAKE_LIB}_nomatmul"
else
    cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="${build_capability}" -S .
    CUDA_VERSION=$CUDA_INSTALLED_VERSION make -C . -j$(nproc)
fi

python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel --dist-dir $PIP_WHEEL_DIR

ls -l $PIP_WHEEL_DIR

uv pip install scipy
uv pip install $PIP_WHEEL_DIR/bitsandbytes*.whl

twine upload --verbose $PIP_WHEEL_DIR/bitsandbytes*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
