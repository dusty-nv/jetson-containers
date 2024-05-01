#!/usr/bin/env bash
set -ex

echo "Building AutoGPTQ ${AUTOGPTQ_VERSION}"
 
git clone --branch=v${AUTOGPTQ_BRANCH} --depth=1 https://github.com/PanQiWei/AutoGPTQ.git

cd AutoGPTQ
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd /
rm -rf AutoGPTQ

pip3 install --no-cache-dir --verbose /opt/auto_gptq*.whl
pip3 show auto-gptq && python3 -c 'import auto_gptq'

twine upload --verbose /opt/auto_gptq*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
