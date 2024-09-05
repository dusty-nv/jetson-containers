#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${COBRA_VERSION} --depth=1 --recursive https://github.com/h-zhao1997/cobra /opt/cobra || \
git clone --depth=1 --recursive https://github.com/h-zhao1997/cobra.git /opt/cobra

# Navigate to the directory containing cobra
cd /opt/cobra

git apply /tmp/COBRA/patch.diff
git diff
git status

pip3 install 'numpy<2'

pip3 wheel --no-build-isolation --wheel-dir=/opt/cobra/wheels .
pip3 install --no-cache-dir --verbose /opt/cobra/wheels/cobra*.whl

cd /opt/mamba
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mamba/wheels/cobra*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
