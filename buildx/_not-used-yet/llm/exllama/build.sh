#!/usr/bin/env bash
set -ex

echo "Building ExLlamaV2 ${EXLLAMA_VERSION}"

cd /opt/exllamav2

sed 's|torch.*|torch|g' -i requirements.txt
sed 's|"torch.*"|"torch"|g' -i setup.py
sed 's|\[\"cublas\"\] if windows else \[\]|\[\"cublas\"\]|g' -i setup.py
sed 's|-mavx2||g' -i setup.py
sed 's|-mavx2||g' -i exllamav2/ext.py
sed 's|#define USE_AVX2||g' -i exllamav2/exllamav2_ext/cpp/sampling.cpp

#pip3 wheel --wheel-dir=/opt --verbose .
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd /
#rm -rf exllamav2
ls -ll /opt

pip3 install /opt/exllamav2*.whl
pip3 show exllamav2 && python3 -c 'import exllamav2'

twine upload --verbose /opt/exllamav2*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
