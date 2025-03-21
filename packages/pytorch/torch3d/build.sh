#!/usr/bin/env bash
set -ex
echo "Building pytorch3d ${PYTORCH3D_VERSION}"
   
git clone --branch v${PYTORCH3D_VERSION} --recursive --depth=1 https://github.com/facebookresearch/pytorch3d /opt/pytorch3d || \
git clone --recursive --depth=1 https://github.com/facebookresearch/pytorch3d /opt/pytorch3d 

cd /opt/pytorch3d
pip3 install scikit-image matplotlib imageio plotly opencv-python-contrib
#export TORCH_CUDA_ARCH_LIST="8.7"
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0

export MAX_JOBS=$(nproc)
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/pytorch3d

pip3 install /opt/pytorch3d*.whl
pip3 show pytorch3d && python3 -c 'import pytorch3d; print(pytorch3d.__version__);'

twine upload --verbose /opt/pytorch3d*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
