#!/usr/bin/env bash
# auto_awq
set -ex

echo "Building AutoAWQ ${AUTOAWQ_VERSION} (kernels=${AUTOAWQ_KERNELS_VERSION})"

git clone --branch=v${AUTOAWQ_KERNELS_VERSION} --depth=1 https://github.com/casper-hansen/AutoAWQ_kernels /opt/AutoAWQ_kernels
cd /opt/AutoAWQ_kernels

sed -i \
   -e 's|"torch[^"]*"|"torch"|g' \
   -e 's|+cu{CUDA_VERSION}||' \
   setup.py

echo "COMPUTE_CAPABILITIES: ${COMPUTE_CAPABILITIES}"

python3 setup.py --verbose bdist_wheel --dist-dir /opt/wheels

git clone --branch=v${AUTOAWQ_VERSION} --depth=1 https://github.com/casper-hansen/AutoAWQ /opt/AutoAWQ

cd /opt/AutoAWQ
sed -i \
   -e 's|"torch[^"]*"|"torch"|g' \
   -e 's|"transformers[^"]*"|"transformers"|g' \
   -e 's|"tokenizers[^"]*"|"tokenizers"|g' \
   -e 's|"accelerate[^"]*"|"accelerate"|g' \
   -e 's|+cu{CUDA_VERSION}||' \
   setup.py

python3 setup.py --verbose bdist_wheel --dist-dir /opt/wheels

cd /
rm -rf /opt/AutoAWQ /opt/AutoAWQ_kernels
ls /opt/wheels/autoawq*

pip3 install --no-cache-dir --verbose \
	/opt/wheels/autoawq_kernels*.whl \
	/opt/wheels/autoawq*.whl
	   
pip3 show autoawq
python3 -c 'import awq'

twine upload --skip-existing --verbose /opt/wheels/autoawq-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --skip-existing --verbose /opt/wheels/autoawq_kernels*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"