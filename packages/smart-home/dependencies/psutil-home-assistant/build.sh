#!/usr/bin/env bash
# psutil-home-assistant
set -xo pipefail

git clone --branch=${PSUTIL_HA_VERSION} https://github.com/home-assistant-libs/psutil-home-assistant /opt/psutil-home-assistant

pip3 wheel --wheel-dir=$PIP_WHEEL_DIR --no-deps --verbose /opt/psutil-home-assistant
rm -rf /opt/psutil-home-assistant

twine upload --skip-existing --verbose $PIP_WHEEL_DIR/psutil_home_assistant*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/psutil_home_assistant*.whl
