#!/usr/bin/env bash
set -ex

echo "Building habitat-sim ${HABITAT_SIM_VERSION} (branch=${HABITAT_SIM_BRANCH})"

git clone --branch=${HABITAT_SIM_BRANCH} --depth=1 --recursive https://github.com/facebookresearch/habitat-sim /opt/habitat-sim

cd /opt/habitat-sim

python3 setup.py bdist_wheel --headless --with-cuda --bullet --dist-dir /opt/wheels

ls /opt/wheels
pip3 install --no-cache-dir --verbose /opt/wheels/habitat*.whl
#pip3 show awq && python3 -c 'import awq' && python3 -m awq.entry --help

twine upload --verbose /opt/wheels/habitat*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

