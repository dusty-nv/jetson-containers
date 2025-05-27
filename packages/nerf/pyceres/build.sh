#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
PYCERES_SRC="${PYCERES_SRC:-/opt/pyceres}"

if [ ! -d $PYCERES_SRC ]; then
    echo "Cloning pyceres version ${PYCERES_VERSION}"
    git clone --branch=v${PYCERES_VERSION} --depth=1 --recursive https://github.com/cvg/pyceres $PYCERES_SRC ||
    git clone --depth=1 --recursive https://github.com/cvg/pyceres $PYCERES_SRC
fi

cd $PYCERES_SRC
export MAX_JOBS=$(nproc)

# Build & install the pyceres wheel
pip3 wheel . -w $PIP_WHEEL_DIR --verbose
pip3 install $PIP_WHEEL_DIR/pyceres*.whl

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/pyceres*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
