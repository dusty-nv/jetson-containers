#!/usr/bin/env bash
set -ex
echo "Building torchvision ${TORCHVISION_VERSION}"
   
git clone --branch v${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision /opt/torchvision
cd /opt/torchvision

git checkout v${TORCHVISION_VERSION}
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/torchvision

pip3 install --no-cache-dir --verbose /opt/torchvision*.whl
pip3 show torchvision && python3 -c 'import torchvision; print(torchvision.__version__);'

twine upload --verbose /opt/torchvision*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
