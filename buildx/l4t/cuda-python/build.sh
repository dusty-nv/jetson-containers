#!/usr/bin/env bash
set -ex

echo "Building cuda-python ${CUDA_PYTHON_VERSION}"
   
git clone --branch v${CUDA_PYTHON_VERSION} --depth=1 https://github.com/NVIDIA/cuda-python
cd cuda-python

sed 's|^numpy.=.*|numpy|g' -i requirements.txt
sed 's|^numba.=.*|numba|g' -i requirements.txt

pip3 install -r requirements.txt
python3 setup.py bdist_wheel --verbose --dist-dir /opt

cd ../
rm -rf cuda-python

pip3 install /opt/cuda*.whl
pip3 show cuda-python && python3 -c 'import cuda; print(cuda.__version__)'

twine upload --verbose /opt/cuda*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
