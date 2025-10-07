#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${LIBCOM_VERSION} --depth=1 --recursive https://github.com/bcmi/libcom /opt/libcom  || \
git clone --depth=1 --recursive https://github.com/bcmi/libcom  /opt/libcom

cd /opt/libcom/requirements

export MAX_JOBS=$(nproc)

cd /opt/libcom/requirements
# Create a temporary runtime.txt without opencv_python==4.1.2.30
grep -v 'opencv_python==4.1.2.30' runtime.txt > runtime_tmp.txt

# Install remaining dependencies
uv pip install --upgrade-strategy eager -r runtime_tmp.txt


cd /opt/libcom/libcom/controllable_composition/source/ControlCom/src/taming-transformers
python3 setup.py install

cd /opt/libcom/
# python3 setup.py install


python3 setup.py bdist_wheel --dist-dir=/opt/libcom/wheels
uv pip install /opt/libcom/wheels/libcom*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/libcom/wheels/libcom*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
