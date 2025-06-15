#!/usr/bin/env bash
set -ex

echo "Building FlashInfer ${FLASHINFER_VERSION}"

REPO_URL="https://github.com/flashinfer-ai/flashinfer"
REPO_DIR="/opt/flashinfer"

git clone --recursive --depth=1 --branch=v${FLASHINFER_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR

cd $REPO_DIR
sed -i 's|options={.*| |g' setup.py
echo "Patched $REPO_DIR/setup.py"
cat setup.py
python3 -m pip install --no-cache-dir build setuptools wheel
python3 -m flashinfer.aot
python3 -m build --no-isolation --wheel
# Install AOT wheel
python3 -m pip install dist/flashinfer_python-*.whl

twine upload --verbose dist/flashinfer_python-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
