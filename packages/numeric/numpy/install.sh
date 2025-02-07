set -ex

pip3 install --upgrade --force-reinstall --no-cache-dir --verbose ${NUMPY_PACKAGE}
pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'