#!/usr/bin/env bash
set -ex
echo "Building PyCUDA ${PYCUDA_VERSION}"
   
git clone --branch=${PYCUDA_VERSION} --depth=1 --recursive https://github.com/inducer/pycuda /opt/pycuda
cd /opt/pycuda

python3 setup.py --verbose build_ext --inplace bdist_wheel --dist-dir /opt

cd /opt
rm -rf /opt/pycuda

pip3 install /opt/pycuda*.whl
pip3 show pycuda && python3 -c 'import pycuda; print(pycuda.VERSION_TEXT)'

twine upload --verbose /opt/pycuda*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
