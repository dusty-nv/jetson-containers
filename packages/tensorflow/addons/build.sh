#!/usr/bin/env bash
set -ex

./llvm.sh 17 all

# Clone the repository if it doesn't exist
git clone --branch=v${TENSORFLOW_ADDONS_VERSION} --depth=1 --recursive https://github.com/tensorflow/addons /opt/tensorflow_addons || \
git clone --depth=1 --recursive https://github.com/tensorflow/addons /opt/tensorflow_addons

cd /opt/tensorflow_addons 
pip3 install -r requirements.txt

export TF_NEED_CUDA="1"
python3 configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg /opt/tensorflow_addons/wheels

pip3 install /opt/tensorflow_addons/wheels/tensorflow_addons*.whl

cd /opt/tensorflow_addons
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/tensorflow_addons/wheels/tensorflow-addons*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
