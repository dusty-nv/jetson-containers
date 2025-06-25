#!/usr/bin/env bash
set -ex

echo "Building ExLlamaV3 ${EXLLAMA_VERSION}"

cd /opt/exllamav3

#pip3 wheel --wheel-dir=/opt --verbose .
pip3 install -U -r requirements.txt
pip3 install -U -r requirements_examples.txt
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd /
#rm -rf exllamav3
ls -ll /opt

pip3 install /opt/exllamav3*.whl
pip3 show exllamav3 && python3 -c 'import exllamav3'

twine upload --verbose /opt/exllamav3*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
