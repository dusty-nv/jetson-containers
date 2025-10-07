#!/usr/bin/env bash
set -ex

echo "Building habitat-sim ${HABITAT_SIM_VERSION} (branch=${HABITAT_SIM_BRANCH})"

git clone --branch=${HABITAT_SIM_BRANCH} --depth=1 --recursive https://github.com/facebookresearch/habitat-sim /opt/habitat-sim && \
git clone --depth=1 --recursive https://github.com/facebookresearch/habitat-sim /opt/habitat-sim

cd /opt/habitat-sim

python3 setup.py bdist_wheel --headless --with-cuda --bullet --dist-dir $PIP_WHEEL_DIR

ls $PIP_WHEEL_DIR
uv pip install $PIP_WHEEL_DIR/habitat*.whl
#uv pip show awq && python3 -c 'import awq' && python3 -m awq.entry --help

twine upload --verbose $PIP_WHEEL_DIR/habitat*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

