#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist

git clone --branch=v${KAOLIN_VERSION} --depth=1 --recursive https://github.com/NVIDIAGameWorks/kaolin /opt/kaolin || \
git clone --depth=1 --recursive https://github.com/NVIDIAGameWorks/kaolin /opt/kaolin

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/kaolin

# Set CC-11 and G++-11 as the default
uv pip install --reinstall blinker
uv pip install -r tools/build_requirements.txt
uv pip install -r tools/viz_requirements.txt
uv pip install -r tools/requirements.txt

export IGNORE_TORCH_VER=1
export CUB_HOME=/usr/local/cuda-*/include/

MAX_JOBS=$(nproc) \
uv build --wheel --no-build-isolation . --out-dir /opt/kaolin/wheels --verbose

uv pip install /opt/kaolin/wheels/kaolin*.whl

cd /opt/kaolin

# Optionally upload to a repository using Twine
twine upload --verbose /opt/kaolin/wheels/kaolin*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
