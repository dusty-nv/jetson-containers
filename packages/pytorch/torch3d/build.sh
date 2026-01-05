#!/usr/bin/env bash
set -ex
echo "Building pytorch3d ${PYTORCH3D_VERSION}"

git clone --branch v${PYTORCH3D_VERSION} --recursive --depth=1 https://github.com/facebookresearch/pytorch3d /opt/pytorch3d || \
git clone --recursive --depth=1 https://github.com/facebookresearch/pytorch3d /opt/pytorch3d

cd /opt/pytorch3d
uv pip install scikit-image matplotlib imageio plotly opencv-contrib-python
#export TORCH_CUDA_ARCH_LIST="8.7"
export CUB_HOME=/usr/local/cuda-*/include/
export MAX_JOBS=$(nproc)
if [[ "$CUDA_VERSION" == "13.0" ]]; then
  export NVCC_FLAGS="-static-global-template-stub=false"
fi
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/pytorch3d

uv pip install /opt/pytorch3d*.whl
uv pip show pytorch3d && python3 -c 'import pytorch3d; print(pytorch3d.__version__);'

twine upload --verbose /opt/pytorch3d*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
