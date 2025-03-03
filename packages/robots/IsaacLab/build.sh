#!/usr/bin/env bash
set -ex

echo "Building IsaacLab ${ISAACLAB_VERSION} (branch=${ISAACLAB_BRANCH})"

git clone --branch=${ISAACLAB_BRANCH} --depth=1 --recursive https://github.com/isaac-sim/IsaacLab /opt/isaaclab  || \
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacLab /opt/isaaclab

cd /opt/isaaclab

pip3 install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com

chmod +x isaaclab.sh
./isaaclab.sh --install # or "./isaaclab.sh -i"

pip3 wheel --wheel-dir=$PIP_WHEEL_DIR --verbose . $PIP_WHEEL_DIR

ls $PIP_WHEEL_DIR

twine upload --verbose $PIP_WHEEL_DIR/isaacsim*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

