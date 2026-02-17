#!/usr/bin/env bash
set -ex

# Clone sources from git
echo "Building pymeshlab ${PYMESHLAB_VERSION}"
PYMESHLAB_SRC="${PYMESHLAB_SRC:-/opt/pymeshlab}"

git clone --branch=v${PYMESHLAB_VERSION}  --depth=1 --recursive https://github.com/cnr-isti-vclab/PyMeshLab $PYMESHLAB_SRC || \
git clone --depth=1 --recursive https://github.com/cnr-isti-vclab/PyMeshLab $PYMESHLAB_SRC

cd $PYMESHLAB_SRC

# Check if make_wheel.sh exists, and use it if available
if [ -f scripts/Linux/make_wheel.sh ]; then
    echo "Using make_wheel.sh to create the Python wheel"
    sh scripts/Linux/make_wheel.sh -w $PIP_WHEEL_DIR
else
    echo "make_wheel.sh not found, using uv build --wheel as fallback"
    echo "Deploying for pymeshlab ${PYMESHLAB_VERSION}"
    sh scripts/Linux/2_deploy.sh
    uv build --wheel . --out-dir $PIP_WHEEL_DIR
fi

# Install and upload python wheel
uv pip install $PIP_WHEEL_DIR/pymeshlab*.whl
twine upload --verbose $PIP_WHEEL_DIR/pymeshlab*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
