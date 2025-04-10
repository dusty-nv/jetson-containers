#!/usr/bin/env bash
set -ex

echo "Building AutoGPTQ ${AUTOGPTQ_VERSION}"
 
git clone --branch=v${AUTOGPTQ_BRANCH} --depth=1 https://github.com/ModelCloud/GPTQModel|| \
git clone --depth=1 https://github.com/ModelCloud/GPTQModel

cd AutoGPTQ
python3 setup.py --verbose bdist_wheel --dist-dir $PIP_WHEEL_DIR

cd /
rm -rf AutoGPTQ

pip3 install $PIP_WHEEL_DIR/gptqmodel*.whl
pip3 show auto-gptq && python3 -c 'import gptqmodel'

twine upload --verbose $PIP_WHEEL_DIR/gptqmodel*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
