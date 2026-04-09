#!/usr/bin/env bash
set -ex

uv pip install pre-commit nanobind==2.5.0
# Clone the repository if it doesn't exist
git clone --branch=${MISTRAL_COMMON_BRANCH} --recursive --depth=1 https://github.com/mistralai/mistral-common /opt/mistral_common ||
git clone --recursive --depth=1 https://github.com/mistralai/mistral-common /opt/mistral_common

cd /opt/mistral_common
uv pip install -U tiktoken
uv build --wheel --no-build-isolation -v --out-dir /opt/mistral_common/wheels .
uv pip install --extra-index-url https://pypi.org/simple/ /opt/mistral_common/wheels/mistral_common*.whl

cd /opt/mistral_common

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mistral_common/wheels/mistral_common*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
