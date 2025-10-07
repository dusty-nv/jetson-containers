#!/usr/bin/env bash
# psutil-home-assistant
set -xo pipefail

git clone --branch=${PSUTIL_HA_BRANCH} https://github.com/home-assistant-libs/psutil-home-assistant /opt/psutil-home-assistant

uv build --wheel --out-dir $PIP_WHEEL_DIR --no-deps --verbose /opt/psutil-home-assistant
uv pip install $PIP_WHEEL_DIR/psutil_home_assistant*.whl
rm -rf /opt/psutil-home-assistant

twine upload --verbose $PIP_WHEEL_DIR/psutil_home_assistant*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

rm $PIP_WHEEL_DIR/psutil_home_assistant*.whl
