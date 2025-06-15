#!/usr/bin/env bash
set -ex

echo "Building IsaacSim ${ISAACSIM_VERSION}"

git clone --branch=${ISAACSIM_VERSION} --depth=1 --recursive https://github.com/isaac-sim/IsaacSim /opt/IsaacSim  || \
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacSim /opt/IsaacSim

cd /opt/IsaacSim

pip3 wheel --wheel-dir=$PIP_WHEEL_DIR --verbose . $PIP_WHEEL_DIR

ls $PIP_WHEEL_DIR

twine upload --verbose $PIP_WHEEL_DIR/isaacsim*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

