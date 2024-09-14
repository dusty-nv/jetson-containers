#!/usr/bin/env bash

set -e
cd /opt/tensorflow-text/

./oss_scripts/run_build.sh

pip3 install --verbose --no-cache-dir /opt/tensorflow-text/tensorflow_text-*.whl

twine upload --verbose /opt/tensorflow-text/tensorflow_text-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
