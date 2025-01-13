#!/usr/bin/env bash
set -ex

# Installation following XGrammar docs: https://xgrammar.mlc.ai/docs/start/install.html#option-2-build-from-source
# Install pip dependencies
pip install pybind11 pre-commit

# Clone the repository if it doesn't exist
git clone --branch=v${XGRAMMAR_VERSION} --recursive --depth=1 https://github.com/mlc-ai/xgrammar.git /opt/xgrammar || 
git clone --recursive --depth=1 https://github.com/mlc-ai/xgrammar.git /opt/xgrammar

# Build and install
cd /opt/xgrammar
pre-commit install
mkdir build
cd build
cmake .. -G Ninja
ninja

# Create the wheel
cd ../python
python3 setup.py bdist_wheel --dist-dir ../wheels

# Install the wheel
# Warning: version number is 0.1.5 even if actual version is 0.1.8, or 0.1.9 due to version.py not being adapted yet: https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/version.py
pip install /opt/xgrammar/wheels/xgrammar*.whl

# Optionally upload to a repository using Twine
# twine upload --verbose /opt/xgrammar/wheels/xgrammar*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
