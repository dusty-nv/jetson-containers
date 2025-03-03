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

python3 setup.py --verbose bdist_wheel --dist-dir $PIP_WHEEL_DIR
pip3 install $PIP_WHEEL_DIR/flashinfer*.whl

twine upload --verbose $PIP_WHEEL_DIR/flashinfer*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
