#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
if [ ! -d /opt/pyceres ]; then
    echo "Cloning pyceres version ${PYCERES_VERSION}"
    git clone --branch=v${PYCERES_VERSION} --depth=1 --recursive https://github.com/cvg/pyceres /opt/pyceres
fi

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/pyceres
pip3 wheel . -w /opt/pyceres/wheels/ 

# Verify the contents of the /opt directory
ls /opt/pyceres/wheels

# Return to the root directory
cd /

pip3 install --no-cache-dir --verbose /opt/pyceres/wheels/pyceres*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/pyceres/wheels/pyceres*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
