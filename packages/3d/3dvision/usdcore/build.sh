#!/usr/bin/env bash
set -ex

echo "Building usd_core ${USD_CORE_VERSION}"

REPO_URL="https://github.com/PixarAnimationStudios/OpenUSD"
REPO_DIR="/opt/usd_core"

git clone --recursive --depth=1 --branch=v${usd_core_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR

cd $REPO_DIR
python3 $REPO_DIR/build_scripts/build_usd.py \
    --build-args="-j $(nproc)" \
    --build-dir=/opt/usd_core/wheels/
# Install AOT wheel
python3 -m pip install /opt/usd_core/wheels/usd_core-*.whl

twine upload --verbose /opt/usd_core/wheels/usd_core-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
