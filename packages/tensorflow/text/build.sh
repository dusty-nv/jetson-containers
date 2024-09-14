#!/usr/bin/env bash

set -e

git clone --recursive --branch=v${TENSORFLOW_VERSION} --depth=1 https://github.com/tensorflow/text /opt/tensorflow-text  || \
git clone --depth=1 --recursive https://github.com/tensorflow/text /opt/tensorflow-text

cd /opt/tensorflow-text/

./oss_scripts/configure.sh
./oss_scripts/prepare_tf_dep.sh
./oss_scripts/run_build.sh

pip3 install --verbose --no-cache-dir /opt/tensorflow-text/tensorflow_text-*.whl

twine upload --verbose /opt/tensorflow-text/tensorflow_text-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
