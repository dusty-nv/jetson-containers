#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${OPENEXR_VERSION} --depth=1 --recursive https://github.com/AcademySoftwareFoundation/openexr /opt/openexr || \
git clone --depth=1 --recursive https://github.com/AcademySoftwareFoundation/openexr /opt/openexr

cd /opt/openexr

pip3 install scikit_build_core cmake pybind11

pip3 wheel --no-build-isolation --wheel-dir=/opt/openexr/wheels .
pip3 install /opt/openexr/wheels/openexr*.whl

cd /opt/openexr


# Optionally upload to a repository using Twine
twine upload --verbose /opt/openexr/wheels/openexr*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
