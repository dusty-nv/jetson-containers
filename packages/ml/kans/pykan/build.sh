#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${PYKAN_VERSION} --depth=1 --recursive https://github.com/KindXiaoming/pykan /opt/pykan  || \
git clone --depth=1 --recursive https://github.com/KindXiaoming/pykan  /opt/pykan

# Navigate to the directory containing pykan's setup.py
cd /opt/pykan

# make the change directly in requirements.txt (GNU/BSD sed)
sed -i 's/==/>=/g' requirements.txt

uv pip install -U -r requirements.txt

python3 setup.py bdist_wheel --dist-dir=/opt/pykan/wheels
uv pip install /opt/pykan/wheels/pykan*.whl

cd /opt/pykan

# Optionally upload to a repository using Twine
twine upload --verbose /opt/pykan/wheels/pykan*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
