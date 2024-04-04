#!/usr/bin/env bash
set -ex

echo "Building AutoAWQ ${AUTOAWQ_VERSION} (kernels=${AUTOAWQ_KERNELS_VERSION})"

git clone --branch=v${AUTOAWQ_KERNELS_VERSION} --depth=1 https://github.com/casper-hansen/AutoAWQ_kernels /opt/AutoAWQ_kernels
cd /opt/AutoAWQ_kernels

echo "AUTOAWQ_CUDA_ARCH: ${AUTOAWQ_CUDA_ARCH}"
sed "s|{75, 80, 86, 89, 90}|{${AUTOAWQ_CUDA_ARCH}}|g" -i setup.py

python3 setup.py --verbose bdist_wheel --dist-dir /opt

git clone --branch=v${AUTOAWQ_VERSION} --depth=1 https://github.com/casper-hansen/AutoAWQ /opt/AutoAWQ

cd /opt/AutoAWQ
sed -i \
   -e 's|"torch>=*"|"torch"|g' \
   -e 's|"transformers>=*",|"transformers"|g' \
   -e 's|"tokenizers>=*",|"tokenizers"|g' \
   -e 's|"accelerate>=*",|"accelerate"|g' \
   setup.py
	   
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd /
rm -rf /opt/AutoAWQ /opt/AutoAWQ_kernels
ls /opt/autoawq*

pip3 install --no-cache-dir --verbose \
	/opt/autoawq_kernels*.whl \
	/opt/autoawq*.whl
	   
pip3 show autoawq && python3 -c 'import awq'

twine upload --verbose /opt/autoawq-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/autoawq_kernels*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"