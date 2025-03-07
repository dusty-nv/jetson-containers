#!/usr/bin/env bash
# ciso8601
set -xo pipefail

git clone --branch=${CISO8601_VERSION} https://github.com/closeio/ciso8601 /opt/ciso8601
git -C /opt/ciso8601 apply /tmp/ciso8601/patch.diff
git -C /opt/ciso8601 diff

pip3 wheel --wheel-dir=$PIP_WHEEL_DIR --no-deps --verbose /opt/ciso8601
pip3 install  $PIP_WHEEL_DIR/ciso8601*.whl
rm -rf /opt/ciso8601

twine upload --skip-existing --verbose $PIP_WHEEL_DIR/ciso8601*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/ciso8601*.whl