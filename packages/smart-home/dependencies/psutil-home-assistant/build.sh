#!/usr/bin/env bash
# psutil-home-assistant
set -xo pipefail

git clone --branch=${PSUTIL_HA_VERSION} https://github.com/home-assistant-libs/psutil-home-assistant /opt/psutil-home-assistant

pip3 wheel --wheel-dir=/opt/wheels --no-deps --verbose /opt/psutil-home-assistant
rm -rf /opt/psutil-home-assistant

twine upload --verbose /opt/wheels/psutil_home_assistant*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
