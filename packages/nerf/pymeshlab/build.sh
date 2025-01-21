#!/usr/bin/env bash
set -ex

echo "Setting up environment for pymeshlab ${PYMESHLAB_VERSION}"

echo "Building pymeshlab ${PYMESHLAB_VERSION}"

git clone --branch=v${PYMESHLAB_VERSION}  --depth=1 --recursive https://github.com/cnr-isti-vclab/PyMeshLab /opt/pymeshlab || \
git clone --depth=1 --recursive https://github.com/cnr-isti-vclab/PyMeshLab /opt/pymeshlab

cd /opt/pymeshlab

# Check if make_wheel.sh exists, and use it if available
if [ -f scripts/Linux/make_wheel.sh ]; then
    echo "Using make_wheel.sh to create the Python wheel"
    sh scripts/Linux/make_wheel.sh -w /opt/pymeshlab/wheels/
else
    echo "make_wheel.sh not found, using pip3 wheel as fallback"
    echo "Deploying for pymeshlab ${PYMESHLAB_VERSION}"
    sh scripts/Linux/2_deploy.sh
    pip3 wheel . -w /opt/pymeshlab/wheels
fi

pip3 install --no-cache-dir --verbose /opt/pymeshlab/wheels/pymeshlab*.whl

twine upload --verbose /opt/pymeshlab/wheels/pymeshlab*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# run here to test if it works
python3 -c "import pymeshlab; ms = pymeshlab.MeshSet()"

echo "pymeshlab OK\n"

