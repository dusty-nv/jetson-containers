#!/usr/bin/env bash
set -ex

echo " ================ Building bitsandbytes ${BITSANDBYTES_VERSION} ================"

echo "### CUDA_INSTALLED_VERSION: $CUDA_INSTALLED_VERSION" 
echo "### CUDA_MAKE_LIB: $CUDA_MAKE_LIB" 
pip3 uninstall -y bitsandbytes || echo "previous bitsandbytes installation not found"

git clone --depth=1 "https://github.com/$BITSANDBYTES_REPO" /opt/bitsandbytes
cd /opt/bitsandbytes
pip3 install -r requirements-dev.txt
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
cmake -DCOMPUTE_BACKEND=cuda -S .
CUDA_VERSION=$CUDA_INSTALLED_VERSION make -C . -j$(nproc)
# CUDA_VERSION=$CUDA_INSTALLED_VERSION make -C . -j$(nproc) "${CUDA_MAKE_LIB}_nomatmul"

cd /opt/bitsandbytes
python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel --dist-dir /opt/wheels/ 
ls -l /opt/wheels/  
pip3 install --no-cache-dir --verbose scipy 
pip3 install --no-cache-dir --verbose /opt/wheels/bitsandbytes*.whl

twine upload --verbose /opt/wheels/bitsandbytes*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
