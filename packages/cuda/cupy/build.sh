#!/usr/bin/env bash
set -ex

echo "Building CuPy ${CUPY_VERSION}"
   
git clone --branch ${CUPY_VERSION} --depth 1 --recursive https://github.com/cupy/cupy cupy
cd cupy

pip3 install fastrlock
python3 setup.py bdist_wheel --verbose
cp dist/cupy*.whl /opt

cd ../
rm -rf cupy

pip3 install /opt/cupy*.whl
pip3 show cupy && python3 -c 'import cupy; print(cupy.show_config())'

twine upload --verbose /opt/cupy*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
