#!/usr/bin/env bash
# auto_awq
set -ex

echo "Building AutoAWQ ${AUTOAWQ_VERSION} (kernels=${AUTOAWQ_KERNELS_VERSION})"

git clone --branch=v${AUTOAWQ_KERNELS_VERSION} --depth=1 https://github.com/casper-hansen/AutoAWQ_kernels /opt/AutoAWQ_kernels ||
git clone --depth=1 https://github.com/casper-hansen/AutoAWQ_kernels

cd /opt/AutoAWQ_kernels

sed -i \
   -e 's|"torch[^"]*"|"torch"|g' \
   -e 's|+cu{CUDA_VERSION}||' \
   setup.py

echo "COMPUTE_CAPABILITIES: ${COMPUTE_CAPABILITIES}"

python3 setup.py --verbose bdist_wheel --dist-dir $PIP_WHEEL_DIR

git clone --branch=v${AUTOAWQ_VERSION} --depth=1 https://github.com/casper-hansen/AutoAWQ /opt/AutoAWQ ||
git clone --depth=1 https://github.com/casper-hansen/AutoAWQ /opt/AutoAWQ

cd /opt/AutoAWQ
sed -i \
   -e 's|"torch[^"]*"|"torch"|g' \
   -e 's|"transformers[^"]*"|"transformers"|g' \
   -e 's|"tokenizers[^"]*"|"tokenizers"|g' \
   -e 's|"accelerate[^"]*"|"accelerate"|g' \
   -e 's|+cu{CUDA_VERSION}||' \
   setup.py

python3 setup.py --verbose bdist_wheel --dist-dir $PIP_WHEEL_DIR

cd /
rm -rf /opt/AutoAWQ /opt/AutoAWQ_kernels
ls $PIP_WHEEL_DIR/autoawq*

uv pip install \
	$PIP_WHEEL_DIR/autoawq_kernels*.whl \
	$PIP_WHEEL_DIR/autoawq*.whl

uv pip show autoawq
python3 -c 'import awq'

twine upload --skip-existing --verbose $PIP_WHEEL_DIR/autoawq-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --skip-existing --verbose $PIP_WHEEL_DIR/autoawq_kernels*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
