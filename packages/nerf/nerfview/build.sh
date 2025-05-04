#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${NERFVIEW_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/nerfview /opt/nerfview || \
git clone --depth=1 --recursive https://github.com/nerfstudio-project/nerfview /opt/nerfview

cd /opt/nerfview

sed -i 's/==/>=/g' pyproject.toml

export BUILD_NO_CUDA=0
export WITH_SYMBOLS=0
export LINE_INFO=1
MAX_JOBS=$(nproc) \
pip3 wheel . -w /opt/nerfview/wheels --verbose

pip3 install /opt/nerfview/wheels/nerfview*.whl

cd /opt/nerfview

# Optionally upload to a repository using Twine
twine upload --verbose /opt/nerfview/wheels/nerfview*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
