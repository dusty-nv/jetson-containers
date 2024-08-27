#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
if [ ! -d /opt/nerfstudio ]; then
    echo "Cloning Nerfstudio version ${NERFSTUDIO_VERSION}"
    git clone --branch=v${NERFSTUDIO_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/nerfstudio /opt/nerfstudio
fi

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/pymeshlab

# Set the maximum number of jobs to the number of available cores
MAX_JOBS=$(nproc)

# Build the wheel, this should trigger the C++ compilation
python3 setup.py build_ext --inplace -j${MAX_JOBS}   # Ensure the C++ code is built
python3 setup.py bdist_wheel --dist-dir /opt         # Create the wheel package

# Verify the contents of the /opt directory
ls /opt

# Return to the root directory
cd /

# Install the newly created wheel package
pip3 install --no-cache-dir --verbose /opt/pymeshlab*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/pymeshlab*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
