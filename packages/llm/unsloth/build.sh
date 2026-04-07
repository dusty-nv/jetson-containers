#!/usr/bin/env bash
set -ex

uv pip install nanobind wheel 'setuptools>=80.9.0' 'setuptools-scm>=9.2.0' twine

git clone --branch=${UNSLOTH_BRANCH} --recursive --depth=1 https://github.com/unslothai/unsloth /opt/unsloth ||
git clone --recursive --depth=1 https://github.com/unslothai/unsloth /opt/unsloth

cd /opt/unsloth

export MAX_JOBS=$(nproc)

uv build --wheel --no-build-isolation -v --out-dir /opt/unsloth/wheels .
uv pip install /opt/unsloth/wheels/unsloth*.whl

sed -i 's/==/>=/g' pyproject.toml && \
sed -i 's/~=/>=/g' pyproject.toml

cd /opt/unsloth
uv pip install compressed-tensors 'unsloth_zoo>=2026.4.3' cut_cross_entropy

twine upload --verbose /opt/unsloth/wheels/unsloth*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
