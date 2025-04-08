#!/usr/bin/env bash
set -ex

echo "Building ExLlamaV3 ${EXLLAMA_VERSION}"

cd /opt/exllamav3

sed 's|torch.*|torch|g' -i requirements.txt
sed 's|"torch.*"|"torch"|g' -i setup.py
sed 's|\[\"cublas\"\] if windows else \[\]|\[\"cublas\"\]|g' -i setup.py
sed 's|-mavx2||g' -i setup.py
sed 's|-mavx2||g' -i exllamav3/ext.py
sed 's|#define USE_AVX2||g' -i exllamav3/exllamav3_ext/cpp/sampling.cpp

#pip3 wheel --wheel-dir=/opt --verbose .
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd /
#rm -rf exllamav3
ls -ll /opt

pip3 install /opt/exllamav3*.whl
pip3 show exllamav3 && python3 -c 'import exllamav3'

twine upload --verbose /opt/exllamav3*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
