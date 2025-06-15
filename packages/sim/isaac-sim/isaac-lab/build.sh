#!/usr/bin/env bash
set -ex

echo "Building IsaacLab ${ISAACLAB_VERSION}"

git clone --branch=${ISAACLAB_VERSION} --depth=1 --recursive https://github.com/isaac-sim/IsaacLab /opt/IsaacLab  || \
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacLab /opt/IsaacLab 

cd /opt/IsaacLab
chmod +x isaaclab.sh
./isaaclab.sh --install # or "./isaaclab.sh -i"

pip3 wheel --wheel-dir=$PIP_WHEEL_DIR --verbose . $PIP_WHEEL_DIR

ls $PIP_WHEEL_DIR

twine upload --verbose $PIP_WHEEL_DIR/isaac*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

