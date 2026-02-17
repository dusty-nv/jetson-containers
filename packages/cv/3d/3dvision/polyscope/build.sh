#!/usr/bin/env bash
set -ex

echo "Building polyscope ${POLYSCOPE_VERSION}"

git clone --branch=v${POLYSCOPE_VERSION} --depth=1 --recursive https://github.com/nmwsharp/polyscope-py /opt/polyscope || \
git clone --depth=1 --recursive https://github.com/nmwsharp/polyscope-py /opt/polyscope

cd /opt/polyscope
export MAX_JOBS=$(nproc)

uv build --wheel . --out-dir $PIP_WHEEL_DIR -v
uv pip install $PIP_WHEEL_DIR/polyscope*.whl

twine upload --verbose $PIP_WHEEL_DIR/polyscope*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
